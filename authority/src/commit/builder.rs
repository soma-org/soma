use std::{
    collections::{BTreeSet, HashSet},
    sync::{Arc, Weak},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tracing::{debug, error, info, instrument};
use types::{
    consensus::ConsensusTransactionKey,
    effects::{TransactionEffects, TransactionEffectsAPI},
    envelope::Message,
    error::SomaResult,
    state_sync::{
        CertifiedCommitSummary, CommitContents, CommitSummary, CommitTimestamp,
        VerifiedCommitSummary,
    },
    transaction::{TransactionKey, TransactionKind},
};

use crate::{
    cache::TransactionCacheRead, epoch_store::AuthorityPerEpochStore,
    handler::SequencedConsensusTransactionKey, state::AuthorityState,
    state_accumulator::StateAccumulator,
};

use super::{output::CommitOutput, CommitStore};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingCommit {
    pub roots: Vec<TransactionKey>,
    pub details: PendingCommitInfo,
}

impl PendingCommit {
    pub fn roots(&self) -> &Vec<TransactionKey> {
        &self.roots
    }

    pub fn details(&self) -> &PendingCommitInfo {
        &self.details
    }

    pub fn height(&self) -> CommitHeight {
        self.details().commit_height
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingCommitInfo {
    pub timestamp_ms: CommitTimestamp,
    pub last_of_epoch: bool,
    pub commit_height: CommitHeight,
}

pub type CommitHeight = u64;

pub struct CommitBuilder {
    state: Arc<AuthorityState>,
    tables: Arc<CommitStore>,
    epoch_store: Arc<AuthorityPerEpochStore>,
    notify: Arc<Notify>,
    notify_aggregator: Arc<Notify>,
    effects_store: Arc<dyn TransactionCacheRead>,
    accumulator: Weak<StateAccumulator>,
    output: Box<dyn CommitOutput>,
}

impl CommitBuilder {
    fn new(
        state: Arc<AuthorityState>,
        tables: Arc<CommitStore>,
        epoch_store: Arc<AuthorityPerEpochStore>,
        notify: Arc<Notify>,
        notify_aggregator: Arc<Notify>,
        effects_store: Arc<dyn TransactionCacheRead>,
        accumulator: Weak<StateAccumulator>,
        output: Box<dyn CommitOutput>,
    ) -> Self {
        Self {
            state,
            tables,
            epoch_store,
            notify,
            notify_aggregator,
            effects_store,
            accumulator,
            output,
        }
    }

    async fn run(mut self) {
        info!("Starting CommitBuilder");
        loop {
            self.maybe_build_commits().await;

            self.notify.notified().await;
        }
    }

    // Instead we need to receive commits from a channel from the consensus layer, then create a new commit with the txs in the commit and the previous commit hash
    // 2) create commits
    // 3) write commits (updates Commit Store)
    // then figure out how to trigger make commit from the consensus layer

    // Groups pending commits until the minimum interval has elapsed, then builds a commit.
    async fn maybe_build_commits(&mut self) {
        // Collect info about the most recently built commit.
        let summary = self
            .epoch_store
            .last_built_commit_summary()
            .expect("epoch should not have ended");
        let mut last_height = summary.clone().and_then(|s| Some(s.0));
        let mut last_timestamp = summary.map(|s| s.1.timestamp_ms);

        let min_commit_interval_ms = 200;

        let mut commits_iter = self
            .epoch_store
            .get_pending_commits(last_height)
            .expect("unexpected epoch store error")
            .into_iter()
            .peekable();
        while let Some((height, pending)) = commits_iter.next() {
            // Group PendingCommits until:
            // - minimum interval has elapsed ...
            let current_timestamp = pending.details().timestamp_ms;
            let can_build = match last_timestamp {
                    Some(last_timestamp) => {
                        current_timestamp >= last_timestamp + min_commit_interval_ms
                    }
                    None => true,
                // - or, next PendingCommit is last-of-epoch (since the last-of-epoch commit
                //   should be written separately) ...
                } || commits_iter
                    .peek()
                    .is_some_and(|(_, next_pending)| next_pending.details().last_of_epoch)
                // - or, we have reached end of epoch.
                    || pending.details().last_of_epoch;

            if !can_build {
                debug!(
                    commit_commit_height = height,
                    ?last_timestamp,
                    ?current_timestamp,
                    "waiting for more PendingCheckpoints: minimum interval not yet elapsed"
                );
                continue;
            }

            // Min interval has elapsed, we can now coalesce and build a commit.
            last_height = Some(height);
            last_timestamp = Some(current_timestamp);
            debug!(
                commit_commit_height = height,
                "Making commit at commit height"
            );
            if let Err(e) = self.make_commit(pending).await {
                error!("Error while making commit, will retry in 1s: {:?}", e);
                tokio::time::sleep(Duration::from_secs(1)).await;
                return;
            }
            // ensure that the task can be cancelled at end of epoch, even if no other await yields
            // execution.
            tokio::task::yield_now().await;
        }
    }

    #[instrument(level = "debug", skip_all, fields(height = pending.details().commit_height))]
    async fn make_commit(&self, pending: PendingCommit) -> anyhow::Result<()> {
        let details = pending.details().clone();

        // Stores the transactions that should be included in the commit. Transactions will be recorded in the commit
        // in this order.
        let txn_in_commit = self.resolve_commit_transactions(pending.roots).await?;

        let new_commit = self.create_commit(txn_in_commit, &details).await?;
        // self.write_commit(details.commit_height, new_commit).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn create_commit(
        &self,
        all_effects: Vec<TransactionEffects>,
        details: &PendingCommitInfo,
    ) -> anyhow::Result<(CommitSummary, CommitContents)> {
        let total = all_effects.len();
        let mut last_commit = self.epoch_store.last_built_commit_summary()?;
        if last_commit.is_none() {
            let epoch = self.epoch_store.epoch();
            if epoch > 0 {
                let previous_epoch = epoch - 1;
                let last_verified = self.tables.get_epoch_last_commit(previous_epoch)?;
                last_commit = last_verified.map(VerifiedCommitSummary::into_summary_and_index);
                if let Some((ref index, _)) = last_commit {
                    debug!("No commits in builder DB, taking commit from previous epoch with index {index}");
                } else {
                    // This is some serious bug with when CheckpointBuilder started so surfacing it via panic
                    panic!("Can not find last commit for previous epoch {previous_epoch}");
                }
            }
        }
        let last_commit_index = last_commit.as_ref().map(|(index, _)| *index);
        info!(
            next_commit_index = last_commit_index.unwrap_or_default() + 1,
            commit_timestamp = details.timestamp_ms,
            "Creating commit(s) for {} transactions",
            all_effects.len(),
        );

        let all_digests: Vec<_> = all_effects
            .iter()
            .map(|effect| *effect.transaction_digest())
            .collect();
        let transactions = self
            .state
            .get_transaction_cache_reader()
            .multi_get_transaction_blocks(&all_digests)?;
        let mut all_fx = Vec::with_capacity(all_effects.len());
        let mut txs = Vec::with_capacity(all_effects.len());
        let mut transaction_keys = Vec::with_capacity(all_effects.len());

        {
            debug!(
                ?last_commit_index,
                "Waiting for {:?} certificates to appear in consensus",
                all_effects.len()
            );

            for (effects, transaction) in all_effects.into_iter().zip(transactions.into_iter()) {
                let transaction = transaction
                    .unwrap_or_else(|| panic!("Could not find executed transaction {:?}", effects));
                match transaction.inner().transaction_data().kind() {
                    TransactionKind::ConsensusCommitPrologue(_) => {}
                    _ => {
                        // All other tx should be included in the call to
                        // `consensus_messages_processed_notify`.
                        transaction_keys.push(SequencedConsensusTransactionKey::External(
                            ConsensusTransactionKey::Certificate(*effects.transaction_digest()),
                        ));
                    }
                }
                txs.push(transaction);
                all_fx.push(effects);
            }

            self.epoch_store
                .consensus_messages_processed_notify(transaction_keys)
                .await?;
        }

        let epoch = self.epoch_store.epoch();

        let first_commit_of_epoch = last_commit
            .as_ref()
            .map(|(_, c)| c.epoch != epoch)
            .unwrap_or(true);

        let last_commit_of_epoch = details.last_of_epoch;

        let index = last_commit
            .as_ref()
            .map(|(_, c)| c.index + 1)
            .unwrap_or_default();
        let timestamp_ms = details.timestamp_ms;
        if let Some((_, last_commit)) = &last_commit {
            if last_commit.timestamp_ms > timestamp_ms {
                error!(
                    "Unexpected decrease of commit timestamp, index: {}, previous: {}, current: {}",
                    index, last_commit.timestamp_ms, timestamp_ms
                );
            }
        }

        let mut effects = all_fx.into_iter();

        // let end_of_epoch_data = if last_commit_of_epoch {
        //     let system_state_obj = self
        //         .augment_epoch_last_commit(
        //             &epoch_rolling_gas_cost_summary,
        //             timestamp_ms,
        //             &mut effects,
        //             &mut signatures,
        //             index,
        //         )
        //         .await?;

        //     let committee = system_state_obj
        //         .get_current_epoch_committee()
        //         .committee()
        //         .clone();

        //     // This must happen after the call to augment_epoch_last_commit,
        //     // otherwise we will not capture the change_epoch tx.
        //     let root_state_digest = {
        //         let state_acc = self
        //             .accumulator
        //             .upgrade()
        //             .expect("No commits should be getting built after local configuration");
        //         let acc =
        //             state_acc.accumulate_commit(effects.clone(), index, &self.epoch_store)?;
        //         state_acc
        //             .accumulate_running_root(&self.epoch_store, index, Some(acc))
        //             .await?;
        //     };

        //     info!("Commit {epoch} root state hash digest: {root_state_digest:?}");

        //     Some(EndOfEpochData {
        //         next_epoch_committee: committee.voting_rights,
        //         next_epoch_protocol_version: ProtocolVersion::new(
        //             system_state_obj.protocol_version(),
        //         ),
        //         epoch_commitments,
        //     })
        // } else {
        //     None
        // };

        let contents =
            CommitContents::new_with_digests(effects.map(|e| e.transaction_digest().clone()));

        let previous_digest = last_commit.as_ref().map(|(_, c)| c.digest());

        let summary = CommitSummary::new(
            epoch,
            index,
            &contents,
            previous_digest,
            // end_of_epoch_data,
            timestamp_ms,
        );

        // if last_commit_of_epoch {
        //     info!(
        //         commit_index = index,
        //         "creating last commit of epoch {}", epoch
        //     );
        // }

        Ok((summary, contents))
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_commit(
        &self,
        height: CommitHeight,
        new_commit: (CertifiedCommitSummary, CommitContents),
    ) -> SomaResult {
        let (summary, contents) = &new_commit;

        debug!(
            commit_commit_height = height,
            commit_index = summary.index,
            contents_digest = ?contents.digest(),
            "writing commit",
        );
        let all_tx_digests: Vec<_> = contents.iter().map(|digest| *digest).collect();

        self.output
            .commit_created(summary, &self.epoch_store, &self.tables)
            .await?;

        let index = summary.index;

        self.tables
            .commit_content
            .write()
            .insert(*contents.digest(), contents.clone());

        self.tables
            .locally_computed_commits
            .write()
            .insert(index, summary.data().clone());

        // Durably commit transactions (but not their outputs) to the database.
        // Called before writing a locally built commit to the CommitStore, so that
        // the inputs of the commit cannot be lost.
        // These transactions are guaranteed to be final unless this validator
        // forks (i.e. constructs a commit which will never be certified). In this case
        // some non-final transactions could be left in the database.
        //
        // This is an intermediate solution until we delay commits to the epoch db. After
        // we have done that, crash recovery will be done by re-processing consensus commits
        // and pending_consensus_transactions, and this method can be removed.
        self.state
            .get_cache_commit()
            .persist_transactions(all_tx_digests.as_slice())
            .await;

        if let Some(certified_commit) = self
            .tables
            .certified_commits
            .read()
            .get(new_commit.0.index())
        {
            self.tables.check_for_commit_fork(
                &new_commit.0.data().clone(),
                &certified_commit.clone().into(),
            );
        }

        self.notify_aggregator.notify_one();
        self.epoch_store
            .process_pending_commit(height, new_commit)?;
        Ok(())
    }

    // Given the root transactions of a pending commit, resolve the transactions should be included in
    // the commit, and return them in the order they should be included in the commit.
    #[instrument(level = "debug", skip_all)]
    async fn resolve_commit_transactions(
        &self,
        roots: Vec<TransactionKey>,
    ) -> SomaResult<Vec<TransactionEffects>> {
        let root_digests = self
            .epoch_store
            .notify_read_executed_digests(&roots)
            .await?;
        let root_effects = self
            .effects_store
            .notify_read_executed_effects(&root_digests)
            .await?;

        // TODO: clean and sort effects
        // let mut sorted: Vec<TransactionEffects> = Vec::with_capacity(unsorted.len() + 1);
        // if let Some((ccp_digest, ccp_effects)) = consensus_commit_prologue {
        //     sorted.push(ccp_effects);
        // }
        // sorted.extend(CausalOrder::causal_sort(unsorted));

        Ok(root_effects)
    }
}
