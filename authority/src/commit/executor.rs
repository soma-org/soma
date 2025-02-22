use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

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
    accumulator::{Accumulator, CommitIndex},
    consensus::{block::BlockAPI, commit::CommittedSubDag, ConsensusTransactionKind},
    digests::TransactionDigest,
    effects::TransactionEffects,
    error::SomaResult,
    transaction::{
        EndOfEpochTransactionKind, VerifiedCertificate, VerifiedExecutableTransaction,
        VerifiedTransaction,
    },
};

use crate::{
    cache::{ObjectCacheRead, TransactionCacheRead},
    epoch_store::AuthorityPerEpochStore,
    output::ConsensusOutputAPI,
    state::AuthorityState,
    state_accumulator::StateAccumulator,
    tx_manager::TransactionManager,
};

use super::CommitStore;

/// The interval to log commit progress, in # of commits processed.
const COMMIT_PROGRESS_LOG_COUNT_INTERVAL: u32 = 5000;

type CommitExecutionBuffer =
    FuturesOrdered<JoinHandle<(CommittedSubDag, Vec<TransactionDigest>, Option<Accumulator>)>>;

pub struct CommitExecutor {
    mailbox: broadcast::Receiver<CommittedSubDag>,
    commit_store: Arc<CommitStore>,
    state: Arc<AuthorityState>,
    object_cache_reader: Arc<dyn ObjectCacheRead>,
    transaction_cache_reader: Arc<dyn TransactionCacheRead>,
    tx_manager: Arc<TransactionManager>,
    accumulator: Arc<StateAccumulator>,
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
        accumulator: Arc<StateAccumulator>,
    ) -> Self {
        Self {
            mailbox,
            commit_store,
            object_cache_reader: state.get_object_cache_reader().clone(),
            transaction_cache_reader: state.get_transaction_cache_reader().clone(),
            tx_manager: state.transaction_manager().clone(),
            state,
            accumulator,
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

        // Complete epoch after executing the last commit of the epoch
        if let Some(highest_executed) = &highest_executed {
            if epoch_store.epoch() == highest_executed.epoch()
                && highest_executed.is_last_commit_of_epoch()
            {
                // We can arrive at this point if we bump the highest_executed_commit watermark, and then
                // crash before completing reconfiguration.
                info!(index = ?highest_executed.commit_ref.index, "final commit of epoch has already been executed");
                return StopReason::EpochComplete;
            }
        }

        let mut next_to_schedule = highest_executed
            .as_ref()
            .map(|c| c.commit_ref.index + 1)
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
                Some(Ok((commit, tx_digests, commit_acc))) = pending.next() => {
                    self.process_executed_commit(&epoch_store, &commit, &tx_digests, commit_acc).await;
                    highest_executed = Some(commit.clone());
                }

                received = self.mailbox.recv() => match received {
                    Ok(commit) => {
                        // info!(
                        //     index = ?commit.commit_ref.index,
                        //     "Received committed sub dag from state sync"
                        // );
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
        commit: &Option<CommittedSubDag>,
    ) -> bool {
        let cur_epoch = epoch_store.epoch();

        if let Some(commit) = commit {
            if commit.epoch() == cur_epoch {
                if let Some(eoe_block) = commit.get_end_of_epoch_block() {
                    info!(
                        ended_epoch = cur_epoch,
                        last_commit = commit.commit_ref.index,
                        "Reached end of epoch",
                    );

                    let epoch_start_timestamp_ms = eoe_block
                        .end_of_epoch_data()
                        .as_ref()
                        .expect("end of epoch block must have end of epoch data")
                        .next_epoch_start_timestamp_ms;

                    let change_epoch_tx_digest = self
                        .execute_change_epoch_tx(
                            cur_epoch,
                            epoch_store.clone(),
                            commit.clone(),
                            epoch_start_timestamp_ms,
                        )
                        .await;

                    let cache_commit = self.state.get_cache_commit();
                    cache_commit
                        .commit_transaction_outputs(cur_epoch, &[change_epoch_tx_digest])
                        .await
                        .expect("commit_transaction_outputs cannot fail");

                    // For finalizing the commit, we need to pass in all commit
                    // transaction effects. This should be a fast operation

                    // Collect transactions from CommittedSubDag
                    let all_tx_digests: HashSet<_> = commit
                        .transactions()
                        .iter()
                        .flat_map(|(_, authority_transactions)| {
                            authority_transactions
                                .iter()
                                .filter_map(|(_, transaction)| {
                                    if let ConsensusTransactionKind::UserTransaction(cert_tx) =
                                        &transaction.kind
                                    {
                                        Some(*cert_tx.digest())
                                    } else {
                                        None
                                    }
                                })
                        })
                        .collect();

                    let all_tx_digests: Vec<_> = all_tx_digests.into_iter().collect();

                    let effects = self
                        .transaction_cache_reader
                        .notify_read_executed_effects(&all_tx_digests)
                        .await
                        .expect("should get effects");

                    finalize_commit(
                        &self.state,
                        self.object_cache_reader.as_ref(),
                        self.transaction_cache_reader.as_ref(),
                        self.commit_store.clone(),
                        &all_tx_digests,
                        &epoch_store,
                        commit.clone(),
                        self.accumulator.clone(),
                        effects,
                    )
                    .await
                    .expect("Finalizing checkpoint cannot fail");

                    self.commit_store
                        .insert_epoch_last_commit(cur_epoch, commit)
                        .expect("Failed to insert epoch last checkpoint");

                    self.accumulator
                        .accumulate_running_root(&epoch_store, commit.commit_ref.index, None)
                        .await
                        .expect("Failed to accumulate running root");
                    self.accumulator
                        .accumulate_epoch(&epoch_store.clone(), commit.commit_ref.index)
                        .expect("Accumulating epoch cannot fail");

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
        cur_epoch: u64,
        epoch_store: Arc<AuthorityPerEpochStore>,
        commit: CommittedSubDag,
        epoch_start_timestamp_ms: u64,
    ) -> TransactionDigest {
        let next_epoch = cur_epoch + 1;

        let tx = VerifiedTransaction::new_end_of_epoch_transaction(
            EndOfEpochTransactionKind::new_change_epoch(next_epoch, epoch_start_timestamp_ms),
        );
        let change_epoch_tx =
            VerifiedExecutableTransaction::new_system(tx.clone(), epoch_store.epoch());
        let change_epoch_tx_digest = change_epoch_tx.digest();
        info!(
            ?next_epoch,
            ?change_epoch_tx_digest,
            "Creating advance epoch transaction"
        );

        if self
            .state
            .get_transaction_cache_reader()
            .is_tx_already_executed(change_epoch_tx_digest)
            .expect("read cannot fail")
        {
            warn!(
                ?change_epoch_tx_digest,
                "Change epoch transaction already executed"
            );

            return change_epoch_tx_digest.clone();
        }

        self.tx_manager.enqueue(
            vec![change_epoch_tx.clone()],
            &epoch_store,
            Some(commit.commit_ref.index),
        );
        handle_execution_effects(
            &self.state,
            vec![change_epoch_tx_digest.clone()],
            commit.clone(),
            self.commit_store.clone(),
            self.object_cache_reader.as_ref(),
            self.transaction_cache_reader.as_ref(),
            epoch_store.clone(),
            self.tx_manager.clone(),
            self.accumulator.clone(),
        )
        .await;

        info!(
            ?change_epoch_tx_digest,
            "Executed change epoch transaction in state sync"
        );

        change_epoch_tx_digest.clone()
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

        while *next_to_schedule <= latest_synced_commit.commit_ref.index
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
            // Check if this commit belongs to current epoch
            if commit.epoch() > epoch_store.epoch() {
                return;
            }

            self.schedule_commit(commit, pending, epoch_store.clone());
            *next_to_schedule += 1;
        }
    }

    #[instrument(level = "error", skip_all, fields(index = ?commit.commit_ref.index, epoch = ?epoch_store.epoch()))]
    fn schedule_commit(
        &self,
        commit: CommittedSubDag,
        pending: &mut CommitExecutionBuffer,
        epoch_store: Arc<AuthorityPerEpochStore>,
    ) {
        debug!(
            "Scheduling commit {} for execution",
            commit.commit_ref.index
        );

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
        let accumulator = self.accumulator.clone();

        epoch_store.notify_synced_commit(commit.commit_ref.index);

        pending.push_back(tokio::spawn(async move {
            let epoch_store = epoch_store.clone();
            let (tx_digests, commit_acc) = loop {
                match execute_commit(
                    commit.clone(),
                    &state,
                    object_cache_reader.as_ref(),
                    transaction_cache_reader.as_ref(),
                    commit_store.clone(),
                    epoch_store.clone(),
                    tx_manager.clone(),
                    accumulator.clone(),
                )
                .await
                {
                    Err(err) => {
                        error!("Error while executing commit, will retry in 1s: {:?}", err);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                    Ok((tx_digests, commit_acc)) => break (tx_digests, commit_acc),
                }
            };
            (commit, tx_digests, commit_acc)
        }));
    }

    /// Post processing and plumbing after we executed a commit. This function is guaranteed
    /// to be called in the order of commit index.
    #[instrument(level = "debug", skip_all)]
    async fn process_executed_commit(
        &self,
        epoch_store: &AuthorityPerEpochStore,
        commit: &CommittedSubDag,
        all_tx_digests: &[TransactionDigest],
        commit_acc: Option<Accumulator>,
    ) {
        // Commit all transaction effects to disk
        let cache_commit = self.state.get_cache_commit();
        debug!(index = ?commit.commit_ref.index, "committing commit transactions to disk");
        cache_commit
            .commit_transaction_outputs(epoch_store.epoch(), all_tx_digests)
            .await
            .expect("commit_transaction_outputs cannot fail");

        epoch_store
            .handle_committed_transactions(all_tx_digests)
            .expect("cannot fail");

        if !commit.is_last_commit_of_epoch() {
            self.accumulator
                .accumulate_running_root(epoch_store, commit.commit_ref.index, commit_acc)
                .await
                .expect("Failed to accumulate running root");
            self.bump_highest_executed_commit(commit);
        }
    }

    fn bump_highest_executed_commit(&self, commit: &CommittedSubDag) {
        // Ensure that we are not skipping commits at any point
        let index = commit.commit_ref.index;
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
                    .delete_commit(prune_index)
                    .expect("Failed to delete full commit contents");
                self.commit_store
                    .delete_digest_index_mapping(&prune_commit.commit_ref.digest)
                    .expect("Failed to delete contents digest -> index mapping");
            } else {
                // If this is directly after a snapshot restore with skiplisting,
                // this is expected for the first `NUM_SAVED_FULL_COMMIT_CONTENTS`
                // commits.
                debug!("Failed to fetch commit with index {:?}", prune_index);
            }
        }

        self.commit_store
            .update_highest_executed_commit(commit)
            .unwrap();
    }
}

// Logs within the function are annotated with the commit index and epoch,
// from schedule_commit().
#[instrument(level = "debug", skip_all, fields(index = ?commit.commit_ref.index, epoch = ?epoch_store.epoch()))]
async fn execute_commit(
    commit: CommittedSubDag,
    state: &AuthorityState,
    object_cache_reader: &dyn ObjectCacheRead,
    transaction_cache_reader: &dyn TransactionCacheRead,
    commit_store: Arc<CommitStore>,
    epoch_store: Arc<AuthorityPerEpochStore>,
    transaction_manager: Arc<TransactionManager>,
    accumulator: Arc<StateAccumulator>,
) -> SomaResult<(Vec<TransactionDigest>, Option<Accumulator>)> {
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
        epoch_store.clone(),
    );

    let commit_acc = execute_transactions(
        all_tx_digests.clone(),
        executable_txns,
        state,
        object_cache_reader,
        transaction_cache_reader,
        commit_store.clone(),
        epoch_store.clone(),
        transaction_manager,
        accumulator,
        commit,
        prepare_start,
    )
    .await?;

    Ok((all_tx_digests, commit_acc))
}

// // Given a commit, find the end of epoch transaction, if it exists
// fn extract_end_of_epoch_tx(
//     commit: &CommittedSubDag,
//     cache_reader: &dyn TransactionCacheRead,
//     commit_store: Arc<CommitStore>,
//     epoch_store: Arc<AuthorityPerEpochStore>,
// ) -> Option<(TransactionDigest, VerifiedExecutableTransaction)> {
//     // Check each authority's transactions in the commit
//     for (authority_index, authority_transactions) in commit.transactions() {
//         // Check the last transaction from this authority
//         if let Some((_, last_tx)) = authority_transactions.last() {
//             if let ConsensusTransactionKind::UserTransaction(cert_tx) = &last_tx.kind {
//                 // Convert to executable transaction
//                 let certificate = VerifiedCertificate::new_unchecked(*cert_tx.clone());
//                 let executable = VerifiedExecutableTransaction::new_from_certificate(certificate);

//                 // TODO: Check if it's an end of epoch transaction
//                 // if executable.is_end_of_epoch_tx() {
//                 //     return Some((*executable.digest(), executable));
//                 // }
//                 return None;
//             }
//         }
//     }

//     // assert!(change_epoch_tx
//     //     .data()
//     //     .intent_message()
//     //     .value
//     //     .is_end_of_epoch_tx());

//     None
// }

// Given a commit, filter out any already executed transactions and return:
// 1. All transaction digests in the commit
// 2. Only the unexecuted transactions that need to be executed
#[allow(clippy::type_complexity)]
fn get_unexecuted_transactions(
    commit: CommittedSubDag,
    cache_reader: &dyn TransactionCacheRead,
    epoch_store: Arc<AuthorityPerEpochStore>,
) -> (Vec<TransactionDigest>, Vec<VerifiedExecutableTransaction>) {
    // Collect transactions from CommittedSubDag
    let transactions: HashMap<_, _> = commit
        .transactions()
        .iter()
        .flat_map(|(_, authority_transactions)| {
            authority_transactions
                .iter()
                .filter_map(|(_, transaction)| {
                    if let ConsensusTransactionKind::UserTransaction(cert_tx) = &transaction.kind {
                        let digest = *cert_tx.digest();
                        let certificate = VerifiedCertificate::new_unchecked(*cert_tx.clone());
                        let executable =
                            VerifiedExecutableTransaction::new_from_certificate(certificate);
                        Some((digest, executable))
                    } else {
                        None
                    }
                })
        })
        .collect();

    // If you need separate vectors:
    let tx_digests: Vec<_> = transactions.keys().copied().collect();
    let transactions: Vec<_> = transactions.values().cloned().collect();

    // Check which transactions are already executed
    let executed_effects_digests = cache_reader
        .multi_get_executed_effects_digests(&tx_digests)
        .expect("failed to read executed_effects from store");

    // Remove the change epoch transaction so that we can special case its execution.
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

    // Filter out already executed transactions
    let executable_txns: Vec<_> = izip!(transactions.into_iter(), executed_effects_digests.iter())
        .filter_map(|(tx, effects_digest)| match effects_digest {
            None => Some(tx),
            Some(_) => {
                trace!(
                    "Transaction with digest {:?} has already been executed",
                    tx.digest()
                );
                None
            }
        })
        .collect();

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
    accumulator: Arc<StateAccumulator>,
    commit: CommittedSubDag,
    prepare_start: Instant,
) -> SomaResult<Option<Accumulator>> {
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

    if commit.commit_ref.index % COMMIT_PROGRESS_LOG_COUNT_INTERVAL == 0 {
        info!(
            "Commit preparation for execution took {:?}",
            prepare_elapsed
        );
    }

    let exec_start = Instant::now();
    transaction_manager.enqueue(
        executable_txns.clone(),
        &epoch_store,
        Some(commit.commit_ref.index),
    );

    let commit_acc = handle_execution_effects(
        state,
        all_tx_digests,
        commit.clone(),
        commit_store,
        object_cache_reader,
        transaction_cache_reader,
        epoch_store,
        transaction_manager,
        accumulator,
    )
    .await;

    let exec_elapsed = exec_start.elapsed();
    if commit.commit_ref.index % COMMIT_PROGRESS_LOG_COUNT_INTERVAL == 0 {
        info!("Commit execution took {:?}", exec_elapsed);
    }

    Ok(commit_acc)
}

#[instrument(level = "error", skip_all, fields(index = ?commit.commit_ref.index, epoch = ?epoch_store.epoch()))]
async fn handle_execution_effects(
    state: &AuthorityState,

    all_tx_digests: Vec<TransactionDigest>,
    commit: CommittedSubDag,
    commit_store: Arc<CommitStore>,
    object_cache_reader: &dyn ObjectCacheRead,
    transaction_cache_reader: &dyn TransactionCacheRead,
    epoch_store: Arc<AuthorityPerEpochStore>,
    transaction_manager: Arc<TransactionManager>,
    accumulator: Arc<StateAccumulator>,
) -> Option<Accumulator> {
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
                if commit.commit_ref.index <= highest_index {
                    error!(
                        "Re-executing commit {} after higher commit {} has executed!",
                        commit.commit_ref.index, highest_index
                    );
                    continue;
                }
                if commit.commit_ref.index > highest_index + 1 {
                    trace!(
                        "Commit {} is still executing. Highest executed = {}",
                        commit.commit_ref.index,
                        highest_index
                    );
                    continue;
                }
                if !blocking_execution {
                    trace!("Commit {} is next to execute.", commit.commit_ref.index);
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
                // TODO: get missing input if needed
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
                // if no end of epoch commit, we must finalize the commit after executing
                // the change epoch tx, which is done after all other commit execution
                if !commit.is_last_commit_of_epoch() {
                    let commit_acc = finalize_commit(
                        state,
                        object_cache_reader,
                        transaction_cache_reader,
                        commit_store.clone(),
                        &all_tx_digests,
                        &epoch_store,
                        commit.clone(),
                        accumulator.clone(),
                        effects,
                    )
                    .await
                    .expect("Finalizing commit cannot fail");
                    return Some(commit_acc);
                } else {
                    return None;
                }
            }
        }
    }
}

#[instrument(level = "info", skip_all, fields(index = ?commit.commit_ref.index, epoch = ?epoch_store.epoch()))]
async fn finalize_commit(
    state: &AuthorityState,
    object_cache_reader: &dyn ObjectCacheRead,
    transaction_cache_reader: &dyn TransactionCacheRead,
    commit_store: Arc<CommitStore>,
    tx_digests: &[TransactionDigest],
    epoch_store: &Arc<AuthorityPerEpochStore>,
    commit: CommittedSubDag,
    accumulator: Arc<StateAccumulator>,
    effects: Vec<TransactionEffects>,
) -> SomaResult<Accumulator> {
    debug!("finalizing commit");
    epoch_store.insert_finalized_transactions(tx_digests, commit.commit_ref.index)?;

    let commit_acc =
        accumulator.accumulate_commit(effects, commit.commit_ref.index, epoch_store)?;

    Ok(commit_acc)
}
