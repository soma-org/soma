use std::{collections::HashMap, sync::Arc, time::Duration};

use futures::stream::FuturesOrdered;
use itertools::{izip, Either};
use tap::{TapFallible, TapOptional};
use tokio::{
    sync::broadcast::{self, error::RecvError},
    task::JoinHandle,
    time::{timeout, Instant},
};
use tokio_stream::StreamExt;
use tracing::{debug, error, info, instrument, trace, warn};
use types::{
    accumulator::CommitIndex, consensus::commit::CommittedSubDag, digests::TransactionDigest, effects::TransactionEffects, error::SomaResult, state_sync::VerifiedCommitSummary, transaction::{VerifiedExecutableTransaction, VerifiedTransaction}
};

use crate::{
    cache::{ObjectCacheRead, TransactionCacheRead},
    epoch_store::AuthorityPerEpochStore,
    state::AuthorityState,
    tx_manager::TransactionManager,
};

use super::CommitStore;

/// The interval to log commit progress, in # of commits processed.
const COMMIT_PROGRESS_LOG_COUNT_INTERVAL: u32 = 5000;

type CommitExecutionBuffer =
    FuturesOrdered<JoinHandle<(VerifiedCommitSummary, Vec<TransactionDigest>)>>;

pub struct CommitExecutor {
    mailbox: broadcast::Receiver<CommittedSubDag>,
    commit_store: Arc<CommitStore>,
    state: Arc<AuthorityState>,
    object_cache_reader: Arc<dyn ObjectCacheRead>,
    transaction_cache_reader: Arc<dyn TransactionCacheRead>,
    tx_manager: Arc<TransactionManager>,
}

#[derive(PartialEq, Eq, Debug)]
pub enum StopReason {
    EpochComplete,
    // RunWithRangeCondition,
}

impl CommitExecutor {
    pub fn new(
        mailbox: broadcast::Receiver<CommittedSubDag>,
        commit_store: Arc<CommitStore>,
        state: Arc<AuthorityState>,
    ) -> Self {
        Self {
            mailbox,
            commit_store,
            object_cache_reader: state.get_object_cache_reader().clone(),
            transaction_cache_reader: state.get_transaction_cache_reader().clone(),
            tx_manager: state.transaction_manager().clone(),
            state,
        }
    }

    /// Ensure that all commits in the current epoch will be executed.
    /// We don't technically need &mut on self, but passing it to make sure only one instance is
    /// running at one time.
    pub async fn run_epoch(
        &mut self,
        epoch_store: Arc<AuthorityPerEpochStore>,
        // run_with_range: Option<RunWithRange>,
    ) -> StopReason {
        debug!("Commit executor running for epoch {}", epoch_store.epoch(),);

        // Decide the first commit to schedule for execution.
        // If we haven't executed anything in the past, we schedule commit 0.
        // Otherwise we schedule the one after highest executed.
        let mut highest_executed = self.commit_store.get_highest_executed_commit().unwrap();

        // TODO: Complete epoch after executing the last commit of the epoch
        // if let Some(highest_executed) = &highest_executed {
        //     if epoch_store.epoch() == highest_executed.epoch()
        //         && highest_executed.is_last_commit_of_epoch()
        //     {
        //         // We can arrive at this point if we bump the highest_executed_commit watermark, and then
        //         // crash before completing reconfiguration.
        //         info!(index = ?highest_executed.index, "final commit of epoch has already been executed");
        //         return StopReason::EpochComplete;
        //     }
        // }

        let mut next_to_schedule = highest_executed
            .as_ref()
            .map(|c| c.index() + 1)
            .unwrap_or_else(|| {
                // TODO this invariant may no longer hold once we introduce snapshots
                assert_eq!(epoch_store.epoch(), 0);
                0
            });
        let mut pending: CommitExecutionBuffer = FuturesOrdered::new();

        let mut now_time = Instant::now();

        loop {
            // If we have executed the last commit of the current epoch, stop.
            // Note: when we arrive here with highest_executed == the final commit of the epoch,
            // we are in an edge case where highest_executed does not actually correspond to the watermark.
            // The watermark is only bumped past the epoch final commit after execution of the change
            // epoch tx.
            if self
                .check_epoch_last_commit(epoch_store.clone(), &highest_executed)
                .await
            {
                self.commit_store
                    .prune_local_summaries()
                    .tap_err(|e| error!("Failed to prune local summaries: {}", e))
                    .ok();

                // be extra careful to ensure we don't have orphans
                assert!(
                    pending.is_empty(),
                    "Pending commit execution buffer should be empty after processing last commit of epoch",
                );
                debug!(epoch = epoch_store.epoch(), "finished epoch");
                return StopReason::EpochComplete;
            }

            self.schedule_synced_commits(
                &mut pending,
                // next_to_schedule will be updated to the next commit to schedule.
                // This makes sure we don't re-schedule the same commit multiple times.
                &mut next_to_schedule,
                epoch_store.clone(),
            );

            // let panic_timeout = Duration::from_secs(45);
            let warning_timeout = Duration::from_secs(5);

            tokio::select! {
                // Check for completed workers and ratchet the highest_commit_executed
                // watermark accordingly. Note that given that commits are guaranteed to
                // be processed (added to FuturesOrdered) in index order, using FuturesOrdered
                // guarantees that we will also ratchet the watermarks in order.
                Some(Ok((commit, tx_digests))) = pending.next() => {
                    self.process_executed_commit(&epoch_store, &commit, &tx_digests).await;
                    highest_executed = Some(commit.clone());
                }

                received = self.mailbox.recv() => match received {
                    Ok(commit) => {
                        info!(
                            index = ?commit.commit_ref.index,
                            "Received commit summary from state sync"
                        );
                    },
                    Err(RecvError::Lagged(num_skipped)) => {
                        debug!(
                            "Commit Execution Recv channel overflowed with {:?} messages",
                            num_skipped,
                        );
                    }
                    Err(RecvError::Closed) => {
                        panic!("Commit Execution Sender (StateSync) closed channel unexpectedly");
                    },
                },

                _ = tokio::time::sleep(warning_timeout) => {
                    warn!(
                        "Received no new synced commits for {warning_timeout:?}. Next commit to be scheduled: {next_to_schedule}",
                    );
                }

                // _ = panic_timeout
                //             .map(|d| Either::Left(tokio::time::sleep(d)))
                //             .unwrap_or_else(|| Either::Right(futures::future::pending())) => {
                //                 panic!("No new synced commits received for {panic_timeout:?}");
                // },
            }
        }
    }

    /// Check whether `commit` is the last commit of the current epoch. If so,
    /// perform special case logic (execute change_epoch tx,
    /// finalize transactions), then return true.
    pub async fn check_epoch_last_commit(
        &self,
        epoch_store: Arc<AuthorityPerEpochStore>,
        commit: &Option<VerifiedCommitSummary>,
    ) -> bool {
        let cur_epoch = epoch_store.epoch();

        if let Some(commit) = commit {
            if commit.epoch() == cur_epoch {
                if let Some((change_epoch_tx_digest, change_epoch_tx)) =
                    extract_end_of_epoch_tx(
                        commit,
                        self.transaction_cache_reader.as_ref(),
                        self.commit_store.clone(),
                        epoch_store.clone(),
                    )
                {

                    info!(
                        ended_epoch = cur_epoch,
                        last_commit = commit.index(),
                        "Reached end of epoch, executing change_epoch transaction",
                    );

                    self.execute_change_epoch_tx(
                        change_epoch_tx_digest,
                        change_epoch_tx,
                        epoch_store.clone(),
                        commit.clone(),
                    )
                    .await;

                    let cache_commit = self.state.get_cache_commit();
                    cache_commit
                        .commit_transaction_outputs(cur_epoch, &[change_epoch_tx_digest])
                        .await
                        .expect("commit_transaction_outputs cannot fail");

                    // For finalizing the commit, we need to pass in all commit
                    // transaction effects, not just the change_epoch tx effects. However,
                    // we have already notify awaited all tx effects separately (once
                    // for change_epoch tx, and once for all other txes). Therefore this
                    // should be a fast operation
                    let all_tx_digests: Vec<_> = self
                        .commit_store
                        .get_commit_contents(&commit.content_digest)
                        .expect("read cannot fail")
                        .expect("Commit contents should exist")
                        .iter()
                        .map(|tx| tx.clone())
                        .collect();

                    let effects = self
                        .transaction_cache_reader
                        .notify_read_executed_effects(all_tx_digests.as_slice())
                        .await
                        .expect("Failed to get executed effects for finalizing commit");

                    finalize_commit(
                        &self.state,
                        self.object_cache_reader.as_ref(),
                        self.transaction_cache_reader.as_ref(),
                        self.commit_store.clone(),
                        &all_tx_digests,
                        &epoch_store,
                        commit.clone(),
                        effects,
                    )
                    .await
                    .expect("Finalizing commit cannot fail");

                    self.commit_store
                        .insert_epoch_last_commit(cur_epoch, commit)
                        .expect("Failed to insert epoch last commit");

                    self.bump_highest_executed_commit(commit);

                    return true;
                }
            }
        }
        false
    }

    #[instrument(level = "info", skip_all)]
    async fn execute_change_epoch_tx(
        &self,
        change_epoch_tx_digest: TransactionDigest,
        change_epoch_tx: VerifiedExecutableTransaction,
        epoch_store: Arc<AuthorityPerEpochStore>,
        commit: VerifiedCommitSummary,
    ) {

        // if change_epoch_tx.contains_shared_object() {
        //     epoch_store
        //         .acquire_shared_locks_from_effects(
        //             &change_epoch_tx,
        //             &change_epoch_fx,
        //             self.object_cache_reader.as_ref(),
        //         )
        //         .await
        //         .expect("Acquiring shared locks for change_epoch tx cannot fail");
        // }

        self.tx_manager.enqueue(
            vec![change_epoch_tx.clone()],
            &epoch_store,
            Some(*commit.index()),
        );
        handle_execution_effects(
            &self.state,
            vec![change_epoch_tx_digest],
            commit.clone(),
            self.commit_store.clone(),
            self.object_cache_reader.as_ref(),
            self.transaction_cache_reader.as_ref(),
            epoch_store.clone(),
            self.tx_manager.clone(),
        )
        .await;
    }


    #[instrument(level = "debug", skip_all)]
    fn schedule_synced_commits(
        &self,
        pending: &mut CommitExecutionBuffer,
        next_to_schedule: &mut CommitIndex,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) {
        let Some(latest_synced_commit) = self
            .commit_store
            .get_highest_synced_commit()
            .expect("Failed to read highest synced commit")
        else {
            debug!("No commits to schedule, highest synced commit is None",);
            return;
        };

        while *next_to_schedule <= *latest_synced_commit.index()
        // && pending.len() < self.config.commit_execution_max_concurrency
        {
            let commit = self
                .commit_store
                .get_commit_by_index(*next_to_schedule)
                .unwrap()
                .unwrap_or_else(|| {
                    panic!(
                        "Commit index {:?} does not exist in commit store",
                        *next_to_schedule
                    )
                });
            if commit.epoch() > epoch_store.epoch() {
                return;
            }
        }
    }

    #[instrument(level = "error", skip_all, fields(index = ?commit.index(), epoch = ?epoch_store.epoch()))]
    fn schedule_commit(
        &self,
        commit: VerifiedCommitSummary,
        pending: &mut CommitExecutionBuffer,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) {
        debug!("Scheduling commit for execution");

        // Mismatch between node epoch and commit epoch after startup
        // crash recovery is invalid
        let commit_epoch = commit.epoch();
        assert_eq!(
            commit_epoch,
            epoch_store.epoch(),
            "Epoch mismatch after startup recovery. commit epoch: {:?}, node epoch: {:?}",
            commit_epoch,
            epoch_store.epoch(),
        );

        let commit_store = self.commit_store.clone();
        let object_cache_reader = self.object_cache_reader.clone();
        let transaction_cache_reader = self.transaction_cache_reader.clone();
        let tx_manager = self.tx_manager.clone();
        let state = self.state.clone();

        epoch_store.notify_synced_commit(*commit.index());

        pending.push_back(tokio::spawn(async move {
            let epoch_store = epoch_store.clone();
            let tx_digests = loop {
                match execute_commit(
                    commit.clone(),
                    &state,
                    object_cache_reader.as_ref(),
                    transaction_cache_reader.as_ref(),
                    commit_store.clone(),
                    epoch_store.clone(),
                    tx_manager.clone(),
                )
                .await
                {
                    Err(err) => {
                        error!("Error while executing commit, will retry in 1s: {:?}", err);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                    Ok(tx_digests) => break tx_digests,
                }
            };
            (commit, tx_digests)
        }));
    }

    /// Post processing and plumbing after we executed a commit. This function is guaranteed
    /// to be called in the order of commit index.
    #[instrument(level = "debug", skip_all)]
    async fn process_executed_commit(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        commit: &VerifiedCommitSummary,
        all_tx_digests: &[TransactionDigest],
    ) {
        // Commit all transaction effects to disk
        let cache_commit = self.state.get_cache_commit();
        debug!(index = ?commit.index, "committing commit transactions to disk");
        cache_commit
            .commit_transaction_outputs(epoch_store.epoch(), all_tx_digests)
            .await
            .expect("commit_transaction_outputs cannot fail");

        epoch_store
            .handle_committed_transactions(all_tx_digests)
            .expect("cannot fail");

        if !commit.is_last_commit_of_epoch() {
            self.bump_highest_executed_commit(commit);
        }
    }

    fn bump_highest_executed_commit(&self, commit: &VerifiedCommitSummary) {
        // Ensure that we are not skipping commits at any point
        let index = *commit.index();
        debug!("Bumping highest_executed_commit watermark to {index:?}");
        if let Some(prev_highest) = self
            .commit_store
            .get_highest_executed_commit_index()
            .unwrap()
        {
            assert_eq!(prev_highest + 1, index);
        } else {
            assert_eq!(index, 0);
        }
        if index % COMMIT_PROGRESS_LOG_COUNT_INTERVAL == 0 {
            info!("Finished syncing and executing commit {}", index);
        }

        // We store a fixed number of additional FullCommitContents after execution is complete
        // for use in state sync.
        const NUM_SAVED_FULL_COMMIT_CONTENTS: u32 = 5_000;
        if index >= NUM_SAVED_FULL_COMMIT_CONTENTS {
            let prune_index = index - NUM_SAVED_FULL_COMMIT_CONTENTS;
            if let Some(prune_commit) = self
                .commit_store
                .get_commit_by_index(prune_index)
                .expect("Failed to fetch commit")
            {
                self.commit_store
                    .delete_full_commit_contents(prune_index)
                    .expect("Failed to delete full commit contents");
                self.commit_store
                    .delete_contents_digest_index_mapping(&prune_commit.content_digest)
                    .expect("Failed to delete contents digest -> index mapping");
            } else {
                // If this is directly after a snapshot restore with skiplisting,
                // this is expected for the first `NUM_SAVED_FULL_COMMIT_CONTENTS`
                // commits.
                debug!(
                    "Failed to fetch commit with index {:?}",
                    prune_index
                );
            }
        }

        self.commit_store
            .update_highest_executed_commit(commit)
            .unwrap();
    }
}

// Logs within the function are annotated with the commit index and epoch,
// from schedule_commit().
#[instrument(level = "debug", skip_all, fields(index = ?commit.index(), epoch = ?epoch_store.epoch()))]
async fn execute_commit(
    commit: VerifiedCommitSummary,
    state: &AuthorityState,
    object_cache_reader: &dyn ObjectCacheRead,
    transaction_cache_reader: &dyn TransactionCacheRead,
    commit_store: Arc<CommitStore>,
    epoch_store: Arc<AuthorityPerEpochStore>,
    transaction_manager: Arc<TransactionManager>,
) -> SomaResult<Vec<TransactionDigest>> {
    debug!("Preparing commit for execution",);
    let prepare_start = Instant::now();

    // this function must guarantee that all transactions in the commit are executed before it
    // returns. This invariant is enforced in two phases:
    // - First, we filter out any already executed transactions from the commit in
    //   get_unexecuted_transactions()
    // - Second, we execute all remaining transactions.

    let (all_tx_digests, executable_txns) = get_unexecuted_transactions(
        commit.clone(),
        transaction_cache_reader,
        commit_store.clone(),
        epoch_store.clone(),
    );

    execute_transactions(
        all_tx_digests.clone(),
        executable_txns,
        state,
        object_cache_reader,
        transaction_cache_reader,
        commit_store.clone(),
        epoch_store.clone(),
        transaction_manager,
        commit,
        prepare_start,
    )
    .await?;

    Ok(all_tx_digests)
}

// Given a commit, find the end of epoch transaction, if it exists
fn extract_end_of_epoch_tx(
    commit: &VerifiedCommitSummary,
    cache_reader: &dyn TransactionCacheRead,
    commit_store: Arc<CommitStore>,
    epoch_store: Arc<AuthorityPerEpochStore>,
) -> Option<(TransactionDigest, VerifiedExecutableTransaction)> {
    // TODO: commit.end_of_epoch_data.as_ref()?;

    // Last commit must have the end of epoch transaction as the last transaction.

    let commit_index = commit.index();
    let tx_digests = commit_store
        .get_commit_contents(&commit.content_digest)
        .expect("Failed to get commit contents from store")
        .unwrap_or_else(|| {
            panic!(
                "Commit contents for digest {:?} does not exist",
                commit.content_digest
            )
        })
        .into_inner();

    let digest = tx_digests
        .last()
        .expect("Final commit must have at least one transaction");

    let change_epoch_tx = cache_reader
        .get_transaction_block(&digest)
        .expect("read cannot fail");

    let change_epoch_tx = VerifiedExecutableTransaction::new_from_commit(
        (*change_epoch_tx.unwrap_or_else(||
            panic!(
                "state-sync should have ensured that transaction with digest {:?} exists for commit: {commit:?}",
                digest,
            )
        )).clone(),
        epoch_store.epoch(),
        *commit_index,
    );

    // assert!(change_epoch_tx
    //     .data()
    //     .intent_message()
    //     .value
    //     .is_end_of_epoch_tx());

    Some((*digest, change_epoch_tx))
}


// Given a commit, filter out any already executed transactions, then return the remaining
// execution digests, transaction digests, transactions to be executed
// (if any) included in the commit.
#[allow(clippy::type_complexity)]
fn get_unexecuted_transactions(
    commit: VerifiedCommitSummary,
    cache_reader: &dyn TransactionCacheRead,
    commit_store: Arc<CommitStore>,
    epoch_store: Arc<AuthorityPerEpochStore>,
) -> (Vec<TransactionDigest>, Vec<VerifiedExecutableTransaction>) {
    let commit_index = commit.index();
    let full_contents = commit_store
        .get_full_commit_contents_by_index(*commit_index)
        .expect("Failed to get commit contents from store")
        .tap_some(|_| debug!("loaded full commit contents in bulk for index {commit_index}"));

    let tx_digests = commit_store
        .get_commit_contents(&commit.content_digest)
        .expect("Failed to get commit contents from store")
        .unwrap_or_else(|| {
            panic!(
                "Commit contents for digest {:?} does not exist",
                commit.content_digest
            )
        })
        .into_inner();

    let full_contents_txns = full_contents.map(|c| {
        c.into_iter()
            .zip(tx_digests.iter())
            .map(|(txn, digest)| (digest, txn))
            .collect::<HashMap<_, _>>()
    });

    // TODO: Remove the change epoch transaction so that we can special case its execution.
    // commit.end_of_epoch_data.as_ref().tap_some(|_| {
    //     let change_epoch_tx_digest = execution_digests
    //         .pop()
    //         .expect("Final commit must have at least one transaction")
    //         .transaction;

    //     let change_epoch_tx = cache_reader
    //         .get_transaction_block(&change_epoch_tx_digest)
    //         .expect("read cannot fail")
    //         .unwrap_or_else(||
    //             panic!(
    //                 "state-sync should have ensured that transaction with digest {change_epoch_tx_digest:?} exists for commit: {}",
    //                 commit.index()
    //             )
    //         );
    //     assert!(change_epoch_tx.data().intent_message().value.is_end_of_epoch_tx());
    // });

    let executed_effects_digests = cache_reader
        .multi_get_executed_effects_digests(&tx_digests)
        .expect("failed to read executed_effects from store");

    let unexecuted_txns: Vec<_> = izip!(tx_digests.iter(), executed_effects_digests.iter())
        .filter_map(|(tx_digest, effects_digest)| match effects_digest {
            None => Some(*tx_digest),
            Some(_effects_digest) => {
                trace!(
                    "Transaction with digest {:?} has already been executed",
                    tx_digest
                );

                None
            }
        })
        .collect();

    // read remaining unexecuted transactions from store
    let executable_txns: Vec<_> = if let Some(full_contents_txns) = full_contents_txns {
        unexecuted_txns
            .into_iter()
            .map(|tx_digest| {
                let tx = full_contents_txns.get(&tx_digest).unwrap();

                VerifiedExecutableTransaction::new_from_commit(
                    VerifiedTransaction::new_unchecked(tx.clone()),
                    epoch_store.epoch(),
                    *commit_index,
                )
            })
            .collect()
    } else {
        cache_reader
            .multi_get_transaction_blocks(unexecuted_txns.as_slice())
            .expect("Failed to get commit txes from store")
            .into_iter()
            .enumerate()
            .map(|(i, tx)| {
                let tx = tx.unwrap_or_else(||
                    panic!(
                        "state-sync should have ensured that transaction with digest {:?} exists for commit: {commit:?}",
                        unexecuted_txns[i]
                    )
                );
                // TODO: change epoch tx is handled specially in check_epoch_last_commit
                // assert!(!tx.data().intent_message().value.is_end_of_epoch_tx());
                
                VerifiedExecutableTransaction::new_from_commit(
                    Arc::try_unwrap(tx).unwrap_or_else(|tx| (*tx).clone()),
                    epoch_store.epoch(),
                    *commit_index,
                )
            })
            .collect()
    };

    (tx_digests, executable_txns)
}

// Logs within the function are annotated with the commit index and epoch,
// from schedule_commit().
#[instrument(level = "debug", skip_all)]
async fn execute_transactions(
    all_tx_digests: Vec<TransactionDigest>,
    executable_txns: Vec<VerifiedExecutableTransaction>,
    state: &AuthorityState,
    object_cache_reader: &dyn ObjectCacheRead,
    transaction_cache_reader: &dyn TransactionCacheRead,
    commit_store: Arc<CommitStore>,
    epoch_store: Arc<AuthorityPerEpochStore>,
    transaction_manager: Arc<TransactionManager>,
    commit: VerifiedCommitSummary,
    prepare_start: Instant,
) -> SomaResult {
    // for tx in &executable_txns {
    //     if tx.contains_shared_object() {
    //         epoch_store
    //             .acquire_shared_locks_from_effects(
    //                 tx,
    //                 digest_to_effects.get(tx.digest()).unwrap(),
    //                 object_cache_reader,
    //             )
    //             .await?;
    //     }
    // }

    let prepare_elapsed = prepare_start.elapsed();

    if commit.index % COMMIT_PROGRESS_LOG_COUNT_INTERVAL == 0 {
        info!(
            "Commit preparation for execution took {:?}",
            prepare_elapsed
        );
    }

    let exec_start = Instant::now();
    transaction_manager.enqueue(executable_txns.clone(), &epoch_store, Some(*commit.index()));

    handle_execution_effects(
        state,
        all_tx_digests,
        commit.clone(),
        commit_store,
        object_cache_reader,
        transaction_cache_reader,
        epoch_store,
        transaction_manager,
    )
    .await;

    let exec_elapsed = exec_start.elapsed();
    if commit.index % COMMIT_PROGRESS_LOG_COUNT_INTERVAL == 0 {
        info!("Commit execution took {:?}", exec_elapsed);
    }

    Ok(())
}

#[instrument(level = "error", skip_all, fields(index = ?commit.index(), epoch = ?epoch_store.epoch()))]
async fn handle_execution_effects(
    state: &AuthorityState,

    all_tx_digests: Vec<TransactionDigest>,
    commit: VerifiedCommitSummary,
    commit_store: Arc<CommitStore>,
    object_cache_reader: &dyn ObjectCacheRead,
    transaction_cache_reader: &dyn TransactionCacheRead,
    epoch_store: Arc<AuthorityPerEpochStore>,
    transaction_manager: Arc<TransactionManager>,
) {
    let log_timeout_sec = Duration::from_secs(30);
    // Once synced_txns have been awaited, all txns should have effects committed.
    let mut periods = 1;
    // Whether the commit is next to execute and blocking additional executions.
    let mut blocking_execution = false;
    loop {
        let effects_future = transaction_cache_reader.notify_read_executed_effects(&all_tx_digests);

        match timeout(log_timeout_sec, effects_future).await {
            Err(_elapsed) => {
                // Reading this value every timeout should be ok.
                let highest_index = commit_store
                    .get_highest_executed_commit_index()
                    .unwrap()
                    .unwrap_or_default();
                if commit.index <= highest_index {
                    error!(
                        "Re-executing commit {} after higher commit {} has executed!",
                        commit.index, highest_index
                    );
                    continue;
                }
                if commit.index > highest_index + 1 {
                    trace!(
                        "Commit {} is still executing. Highest executed = {}",
                        commit.index,
                        highest_index
                    );
                    continue;
                }
                if !blocking_execution {
                    trace!("Commit {} is next to execute.", commit.index);
                    blocking_execution = true;
                    continue;
                }

                // Only log details when the commit is next to execute, but has not finished
                // execution within log_timeout_sec.
                let missing_digests: Vec<TransactionDigest> = transaction_cache_reader
                    .multi_get_executed_effects_digests(&all_tx_digests)
                    .expect("multi_get_executed_effects cannot fail")
                    .iter()
                    .zip(all_tx_digests.clone())
                    .filter_map(
                        |(fx, digest)| {
                            if fx.is_none() {
                                Some(digest)
                            } else {
                                None
                            }
                        },
                    )
                    .collect();

                if missing_digests.is_empty() {
                    // All effects just become available.
                    continue;
                }

                // Print out more information for the 1st pending transaction, which should have
                // all of its input available.
                let pending_digest = missing_digests.first().unwrap();
                // if let Some(missing_input) = transaction_manager.get_missing_input(pending_digest) {
                //     warn!(
                //         "Transaction {pending_digest:?} has missing input objects {missing_input:?}",
                //     );
                // }
                periods += 1;
            }
            Ok(Err(err)) => panic!("Failed to notify_read_executed_effects: {:?}", err),
            Ok(Ok(effects)) => {
                // TODO: if end of epoch commit, we must finalize the commit after executing
                // the change epoch tx, which is done after all other commit execution
                // if commit.end_of_epoch_data.is_none() {
                //     finalize_commit(
                //         state,
                //         object_cache_reader,
                //         transaction_cache_reader,
                //         commit_store.clone(),
                //         &all_tx_digests,
                //         &epoch_store,
                //         commit.clone(),
                //         effects,
                //     )
                //     .await
                //     .expect("Finalizing commit cannot fail");
                // }
            }
        }
    }
}

#[instrument(level = "info", skip_all, fields(index = ?commit.index(), epoch = ?epoch_store.epoch()))]
async fn finalize_commit(
    state: &AuthorityState,
    object_cache_reader: &dyn ObjectCacheRead,
    transaction_cache_reader: &dyn TransactionCacheRead,
    commit_store: Arc<CommitStore>,
    tx_digests: &[TransactionDigest],
    epoch_store: &Arc<AuthorityPerEpochStore>,
    commit: VerifiedCommitSummary,
    effects: Vec<TransactionEffects>,
) -> SomaResult {
    debug!("finalizing commit");
    epoch_store.insert_finalized_transactions(tx_digests, commit.index)?;

    Ok(())
}
