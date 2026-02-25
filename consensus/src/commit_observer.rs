// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use crate::{
    CommitConsumerArgs,
    commit_finalizer::{CommitFinalizer, CommitFinalizerHandle},
    dag_state::DagState,
    leader_schedule::LeaderSchedule,
    linearizer::Linearizer,
    transaction_certifier::TransactionCertifier,
};
use parking_lot::RwLock;
use tokio::time::Instant;
use tracing::info;
use types::consensus::{
    block::{BlockAPI, VerifiedBlock},
    commit::{CommitAPI, CommittedSubDag, load_committed_subdag_from_store},
    context::Context,
};
use types::error::ConsensusResult;
use types::storage::consensus::Store;

/// Role of CommitObserver
/// - Called by core when try_commit() returns newly committed leaders.
/// - The newly committed leaders are sent to commit observer and then commit observer
///   gets subdags for each leader via the commit interpreter (linearizer)
/// - The committed subdags are sent as consensus output via an unbounded tokio channel.
///
/// There is no flow control on sending output. Consensus backpressure is applied earlier
/// at consensus input level, and on commit sync.
///
/// Commit is persisted in store before the CommittedSubDag is sent to the commit handler.
/// When Sui recovers, it blocks until the commits it knows about are recovered. So consensus
/// must be able to quickly recover the commits it has sent to Sui.
pub(crate) struct CommitObserver {
    context: Arc<Context>,
    dag_state: Arc<RwLock<DagState>>,
    /// Persistent storage for blocks, commits and other consensus data.
    store: Arc<dyn Store>,
    transaction_certifier: TransactionCertifier,
    leader_schedule: Arc<LeaderSchedule>,
    /// Component to deterministically collect subdags for committed leaders.
    commit_interpreter: Linearizer,
    /// Handle to an unbounded channel to send output commits.
    commit_finalizer_handle: CommitFinalizerHandle,
}

impl CommitObserver {
    pub(crate) async fn new(
        context: Arc<Context>,
        commit_consumer: CommitConsumerArgs,
        dag_state: Arc<RwLock<DagState>>,
        transaction_certifier: TransactionCertifier,
        leader_schedule: Arc<LeaderSchedule>,
    ) -> Self {
        let store = dag_state.read().store();
        let commit_interpreter = Linearizer::new(context.clone(), dag_state.clone());
        let commit_finalizer_handle = CommitFinalizer::start(
            context.clone(),
            dag_state.clone(),
            transaction_certifier.clone(),
            commit_consumer.commit_sender.clone(),
        );

        let mut observer = Self {
            context,
            dag_state,
            store,
            transaction_certifier,
            leader_schedule,
            commit_interpreter,
            commit_finalizer_handle,
        };
        observer.recover_and_send_commits(&commit_consumer).await;

        // Recover blocks needed for future commits (and block proposals).
        // Some blocks might have been recovered as committed blocks in recover_and_send_commits().
        // They will just be ignored.
        tokio::runtime::Handle::current()
            .spawn_blocking({
                let transaction_certifier = observer.transaction_certifier.clone();
                let gc_round = observer.dag_state.read().gc_round();
                move || {
                    transaction_certifier.recover_blocks_after_round(gc_round);
                }
            })
            .await
            .expect("Spawn blocking should not fail");

        observer
    }

    /// Creates and returns a list of committed subdags containing committed blocks, from a sequence
    /// of selected leader blocks, and whether they come from local committer or commit sync remotely.
    ///
    /// Also, buffers the commits to DagState and forwards committed subdags to commit finalizer.
    pub(crate) fn handle_commit(
        &mut self,
        committed_leaders: Vec<VerifiedBlock>,
        local: bool,
    ) -> ConsensusResult<Vec<CommittedSubDag>> {
        let mut committed_sub_dags = self.commit_interpreter.handle_commit(committed_leaders);

        // Set if the commit is produced from local DAG, or received through commit sync.
        for subdag in committed_sub_dags.iter_mut() {
            subdag.decided_with_local_blocks = local;
        }

        // Send scores as part of the first sub dag, if the leader schedule has been updated.
        let schedule_updated = self.leader_schedule.leader_schedule_updated(&self.dag_state);
        if schedule_updated {
            let reputation_scores_desc =
                self.leader_schedule.leader_swap_table.read().reputation_scores_desc.clone();
            committed_sub_dags[0].reputation_scores_desc = reputation_scores_desc;
        }

        for commit in committed_sub_dags.iter() {
            tracing::debug!(
                "Sending commit {} leader {} to finalization and execution.",
                commit.commit_ref,
                commit.leader
            );
            tracing::trace!("Committed subdag: {:#?}", commit);
            // Failures in sender.send() are assumed to be permanent
            self.commit_finalizer_handle.send(commit.clone())?;
        }

        self.dag_state.write().add_scoring_subdags(committed_sub_dags.clone());

        Ok(committed_sub_dags)
    }

    async fn recover_and_send_commits(&mut self, commit_consumer: &CommitConsumerArgs) {
        let now = Instant::now();

        let replay_after_commit_index = commit_consumer.replay_after_commit_index;

        let last_commit =
            self.store.read_last_commit().expect("Reading the last commit should not fail");
        let Some(last_commit) = &last_commit else {
            assert_eq!(
                replay_after_commit_index, 0,
                "Commit replay should start at the beginning if there is no commit history"
            );
            info!("Nothing to recover for commit observer - starting new epoch");
            return;
        };

        let last_commit_index = last_commit.index();
        if last_commit_index == replay_after_commit_index {
            info!(
                "Nothing to recover for commit observer - replay is requested immediately after last commit index {last_commit_index}"
            );
            return;
        }
        assert!(last_commit_index > replay_after_commit_index);

        info!(
            "Recovering commit observer in the range [{}..={last_commit_index}]",
            replay_after_commit_index + 1,
        );

        // To avoid scanning too many commits at once and load in memory,
        // we limit the batch size to 250 and iterate over.
        const COMMIT_RECOVERY_BATCH_SIZE: u32 = if cfg!(test) { 3 } else { 250 };

        let mut last_sent_commit_index = replay_after_commit_index;

        // Make sure that there is no pending commits to be written to the store.
        self.dag_state.read().ensure_commits_to_write_is_empty();

        let mut seen_unfinalized_commit = false;
        for start_index in (replay_after_commit_index + 1..=last_commit_index)
            .step_by(COMMIT_RECOVERY_BATCH_SIZE as usize)
        {
            let end_index =
                start_index.saturating_add(COMMIT_RECOVERY_BATCH_SIZE - 1).min(last_commit_index);

            let unsent_commits = self
                .store
                .scan_commits((start_index..=end_index).into())
                .expect("Scanning commits should not fail");
            assert_eq!(
                unsent_commits.len() as u32,
                end_index.checked_sub(start_index).unwrap() + 1,
                "Gap in scanned commits: start index: {start_index}, end index: {end_index}, commits: {:?}",
                unsent_commits,
            );

            // Buffered unsent commits in DAG state which is required to contain them when they are flushed
            // by CommitFinalizer.
            self.dag_state.write().recover_commits_to_write(unsent_commits.clone());

            info!(
                "Recovered {} unsent commits in range [{start_index}..={end_index}]",
                unsent_commits.len()
            );

            // Resend all the committed subdags to the consensus output channel
            // for all the commits above the last processed index.
            for commit in unsent_commits.into_iter() {
                // Commit index must be continuous.
                last_sent_commit_index += 1;
                assert_eq!(commit.index(), last_sent_commit_index);

                // On recovery leader schedule will be updated with the current scores
                // and the scores will be passed along with the last commit of this recovered batch sent to
                // Sui so that the current scores are available for submission.
                let reputation_scores = if commit.index() == last_commit_index {
                    self.leader_schedule.leader_swap_table.read().reputation_scores_desc.clone()
                } else {
                    vec![]
                };

                let committed_sub_dag = load_committed_subdag_from_store(
                    self.store.as_ref(),
                    commit,
                    reputation_scores,
                );

                if !committed_sub_dag.recovered_rejected_transactions {
                    // When the commit has no associated storage entry for rejected transactions,
                    // even if an empty set, the commit is unfinalized.
                    seen_unfinalized_commit = true;
                }

                if seen_unfinalized_commit {
                    // After observing the first unfinalized commit, the rest of recovered commits should all be unfinalized.
                    assert!(!committed_sub_dag.recovered_rejected_transactions);
                    // All unfinalized commit cannot be assumed to be decided with local blocks, because they
                    // might have been received through commit sync.
                    assert!(!committed_sub_dag.decided_with_local_blocks);
                    // All unfinalized commits need to be processed by the CommitFinalizer, making it necessary to
                    // recover and vote on the blocks in this commit.
                    self.transaction_certifier
                        .recover_and_vote_on_blocks(committed_sub_dag.blocks.clone());
                }

                self.commit_finalizer_handle.send(committed_sub_dag).unwrap();

                tokio::task::yield_now().await;
            }
        }

        assert_eq!(
            last_sent_commit_index, last_commit_index,
            "We should have sent all commits up to the last commit {}",
            last_commit_index
        );

        info!(
            "Commit observer recovery [{}..={}] completed, took {:?}",
            replay_after_commit_index + 1,
            last_commit_index,
            now.elapsed()
        );
    }
}

// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
// Adapted for SOMA.
#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use parking_lot::RwLock;
    use tokio::time::timeout;
    use types::committee::AuthorityIndex;
    use types::consensus::{
        block::{BlockAPI, BlockRef},
        commit::{CommitAPI, CommitIndex, CommittedSubDag},
        context::Context,
    };
    use types::storage::consensus::Store;
    use types::storage::consensus::mem_store::MemStore;

    use crate::{
        CommitConsumerArgs,
        block_verifier::NoopBlockVerifier,
        dag_state::DagState,
        leader_schedule::{LeaderSchedule, LeaderSwapTable},
        linearizer::median_timestamp_by_stake,
        test_dag_builder::DagBuilder,
        transaction_certifier::TransactionCertifier,
    };

    use super::CommitObserver;

    #[tokio::test]
    async fn test_handle_commit() {
        let num_authorities = 4;
        let (context, _keys) = Context::new_for_test(num_authorities);
        let context = Arc::new(context);

        let mem_store = Arc::new(MemStore::new());
        let dag_state = Arc::new(RwLock::new(DagState::new(context.clone(), mem_store.clone())));
        let last_processed_commit_index = 0;
        let (commit_consumer, mut commit_receiver, _block_receiver) =
            CommitConsumerArgs::new(0, last_processed_commit_index);
        let (blocks_sender, _blocks_receiver) = tokio::sync::mpsc::unbounded_channel();
        let transaction_certifier = TransactionCertifier::new(
            context.clone(),
            Arc::new(NoopBlockVerifier {}),
            dag_state.clone(),
            blocks_sender,
        );
        const NUM_OF_COMMITS_PER_SCHEDULE: u64 = 5;
        let leader_schedule = Arc::new(
            LeaderSchedule::new(context.clone(), LeaderSwapTable::default())
                .with_num_commits_per_schedule(NUM_OF_COMMITS_PER_SCHEDULE),
        );

        let mut observer = CommitObserver::new(
            context.clone(),
            commit_consumer,
            dag_state.clone(),
            transaction_certifier.clone(),
            leader_schedule.clone(),
        )
        .await;

        // Populate fully connected test blocks for round 0 ~ 10, authorities 0 ~ 3.
        let num_rounds = 10;
        let mut builder = DagBuilder::new(context.clone());
        builder.layers(1..=num_rounds).build().persist_layers(dag_state.clone());
        transaction_certifier
            .add_voted_blocks(builder.all_blocks().iter().map(|b| (b.clone(), vec![])).collect());

        let leaders = builder
            .leader_blocks(1..=num_rounds)
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();

        // Commit first 5 leaders.
        let mut commits = observer.handle_commit(leaders[0..5].to_vec(), true).unwrap();

        // Trigger a leader schedule update.
        leader_schedule.update_leader_schedule_v2(&dag_state);

        // Commit the next 5 leaders.
        commits.extend(observer.handle_commit(leaders[5..].to_vec(), true).unwrap());

        // Check commits are returned by CommitObserver::handle_commit is accurate
        let mut expected_stored_refs: Vec<BlockRef> = vec![];
        for (idx, subdag) in commits.iter().enumerate() {
            tracing::info!("{subdag:?}");
            assert_eq!(subdag.leader, leaders[idx].reference());

            // 5th subdag should contain the updated scores.
            if idx == 5 {
                // With normalized voting power (2500 per authority for 4 authorities),
                // each vote contributes 2500 stake. With 9 votes per authority across
                // 5 committed subdags, scores are 9 * 2500 = 22500.
                let scores = vec![
                    (AuthorityIndex::new_for_test(1), 22500),
                    (AuthorityIndex::new_for_test(3), 22500),
                    (AuthorityIndex::new_for_test(0), 22500),
                    (AuthorityIndex::new_for_test(2), 22500),
                ];
                assert_eq!(subdag.reputation_scores_desc, scores);
            } else {
                assert!(subdag.reputation_scores_desc.is_empty());
            }

            let expected_ts = {
                let block_refs = leaders[idx]
                    .ancestors()
                    .iter()
                    .filter(|block_ref| block_ref.round == leaders[idx].round() - 1)
                    .cloned()
                    .collect::<Vec<_>>();
                let blocks =
                    dag_state.read().get_blocks(&block_refs).into_iter().map(|block_opt| {
                        block_opt.expect("We should have all blocks in dag state.")
                    });
                median_timestamp_by_stake(&context, blocks).unwrap()
            };

            let expected_ts =
                if idx == 0 { expected_ts } else { expected_ts.max(commits[idx - 1].timestamp_ms) };

            assert_eq!(expected_ts, subdag.timestamp_ms);

            if idx == 0 {
                // First subdag includes the leader block plus all ancestor blocks
                // of the leader minus the genesis round blocks
                assert_eq!(subdag.blocks.len(), 1);
            } else {
                // Every subdag after will be missing the leader block from the previous
                // committed subdag
                assert_eq!(subdag.blocks.len(), num_authorities);
            }
            for block in subdag.blocks.iter() {
                expected_stored_refs.push(block.reference());
                assert!(block.round() <= leaders[idx].round());
            }
            assert_eq!(subdag.commit_ref.index, idx as CommitIndex + 1);
        }

        // Check commits sent over consensus output channel is accurate
        let mut processed_subdag_index = 0;
        while let Ok(Some(subdag)) = timeout(Duration::from_secs(1), commit_receiver.recv()).await {
            assert_eq!(subdag, commits[processed_subdag_index]);
            processed_subdag_index = subdag.commit_ref.index as usize;
            if processed_subdag_index == leaders.len() {
                break;
            }
        }
        assert_eq!(processed_subdag_index, leaders.len());

        verify_channel_empty(&mut commit_receiver).await;

        // Check commits have been persisted to storage
        let last_commit = mem_store.read_last_commit().unwrap().unwrap();
        assert_eq!(last_commit.index(), commits.last().unwrap().commit_ref.index);
        let all_stored_commits = mem_store.scan_commits((0..=CommitIndex::MAX).into()).unwrap();
        assert_eq!(all_stored_commits.len(), leaders.len());
    }

    #[tokio::test]
    async fn test_recover_and_send_commits() {
        let num_authorities = 4;
        let context = Arc::new(Context::new_for_test(num_authorities).0);
        let mem_store = Arc::new(MemStore::new());
        let dag_state = Arc::new(RwLock::new(DagState::new(context.clone(), mem_store.clone())));
        let (blocks_sender, _blocks_receiver) = tokio::sync::mpsc::unbounded_channel();
        let transaction_certifier = TransactionCertifier::new(
            context.clone(),
            Arc::new(NoopBlockVerifier {}),
            dag_state.clone(),
            blocks_sender,
        );
        let last_processed_commit_index = 0;
        let (commit_consumer, mut commit_receiver, _block_receiver) =
            CommitConsumerArgs::new(0, last_processed_commit_index);
        let leader_schedule =
            Arc::new(LeaderSchedule::from_store(context.clone(), dag_state.clone()));

        let mut observer = CommitObserver::new(
            context.clone(),
            commit_consumer,
            dag_state.clone(),
            transaction_certifier.clone(),
            leader_schedule.clone(),
        )
        .await;

        // Populate fully connected test blocks for round 0 ~ 10, authorities 0 ~ 3.
        let num_rounds = 10;
        let mut builder = DagBuilder::new(context.clone());
        builder.layers(1..=num_rounds).build().persist_layers(dag_state.clone());
        transaction_certifier
            .add_voted_blocks(builder.all_blocks().iter().map(|b| (b.clone(), vec![])).collect());

        let leaders = builder
            .leader_blocks(1..=num_rounds)
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();

        // Commit first batch of leaders (2) and "receive" the subdags as the
        // consumer of the consensus output channel.
        let expected_last_processed_index: usize = 2;
        let mut commits = observer
            .handle_commit(leaders[..expected_last_processed_index].to_vec(), true)
            .unwrap();

        // Check commits sent over consensus output channel is accurate
        let mut processed_subdag_index = 0;
        while let Ok(Some(subdag)) = timeout(Duration::from_secs(1), commit_receiver.recv()).await {
            tracing::info!("Processed {subdag}");
            assert_eq!(subdag, commits[processed_subdag_index]);
            assert_eq!(subdag.reputation_scores_desc, vec![]);
            processed_subdag_index = subdag.commit_ref.index as usize;
            if processed_subdag_index == expected_last_processed_index {
                break;
            }
        }
        assert_eq!(processed_subdag_index, expected_last_processed_index);

        verify_channel_empty(&mut commit_receiver).await;

        // Check last stored commit is correct
        let last_commit = mem_store.read_last_commit().unwrap().unwrap();
        assert_eq!(last_commit.index(), expected_last_processed_index as CommitIndex);

        // Handle next batch of leaders (10 - 2 = 8), these will be sent by consensus but not
        // "processed" by consensus output channel.
        commits.append(
            &mut observer
                .handle_commit(leaders[expected_last_processed_index..].to_vec(), true)
                .unwrap(),
        );

        let expected_last_sent_index = num_rounds as usize;
        while let Ok(Some(subdag)) = timeout(Duration::from_secs(1), commit_receiver.recv()).await {
            tracing::info!("{subdag} was sent but not processed by consumer");
            assert_eq!(subdag, commits[processed_subdag_index]);
            assert!(subdag.decided_with_local_blocks);
            assert_eq!(subdag.reputation_scores_desc, vec![]);
            processed_subdag_index = subdag.commit_ref.index as usize;
            if processed_subdag_index == expected_last_sent_index {
                break;
            }
        }
        assert_eq!(processed_subdag_index, expected_last_sent_index);

        verify_channel_empty(&mut commit_receiver).await;

        // Check last stored commit is correct.
        let last_commit = mem_store.read_last_commit().unwrap().unwrap();
        assert_eq!(last_commit.index(), expected_last_sent_index as CommitIndex);

        // Replay commits after index 2.
        {
            let replay_after_commit_index = 2;
            let consumer_last_processed_commit_index = 10;
            let dag_state =
                Arc::new(RwLock::new(DagState::new(context.clone(), mem_store.clone())));
            let (commit_consumer, mut commit_receiver, _block_receiver) = CommitConsumerArgs::new(
                replay_after_commit_index,
                consumer_last_processed_commit_index,
            );
            let _observer = CommitObserver::new(
                context.clone(),
                commit_consumer,
                dag_state.clone(),
                transaction_certifier.clone(),
                leader_schedule.clone(),
            )
            .await;

            let mut processed_subdag_index = replay_after_commit_index;
            while let Ok(Some(mut subdag)) =
                timeout(Duration::from_secs(1), commit_receiver.recv()).await
            {
                tracing::info!("Received {subdag} on recovery");
                assert_eq!(subdag.commit_ref.index, processed_subdag_index + 1);
                assert!(subdag.recovered_rejected_transactions);

                // Allow comparison with committed subdag before recovery.
                subdag.recovered_rejected_transactions = false;
                assert_eq!(subdag, commits[processed_subdag_index as usize]);

                assert!(subdag.decided_with_local_blocks);
                assert_eq!(subdag.reputation_scores_desc, vec![]);
                processed_subdag_index = subdag.commit_ref.index;
                if processed_subdag_index == consumer_last_processed_commit_index {
                    break;
                }
            }
            assert_eq!(processed_subdag_index, consumer_last_processed_commit_index);

            verify_channel_empty(&mut commit_receiver).await;
        }

        // Replay commits from index 10, which is the last persisted commit.
        {
            let replay_after_commit_index = 10;
            let consumer_last_processed_commit_index = 10;
            let dag_state =
                Arc::new(RwLock::new(DagState::new(context.clone(), mem_store.clone())));
            let (commit_consumer, mut commit_receiver, _block_receiver) = CommitConsumerArgs::new(
                replay_after_commit_index,
                consumer_last_processed_commit_index,
            );
            let _observer = CommitObserver::new(
                context.clone(),
                commit_consumer,
                dag_state.clone(),
                transaction_certifier.clone(),
                leader_schedule.clone(),
            )
            .await;

            // No commits should be resubmitted
            verify_channel_empty(&mut commit_receiver).await;
        }
    }

    /// After receiving all expected subdags, ensure channel is empty
    async fn verify_channel_empty(
        receiver: &mut tokio::sync::mpsc::UnboundedReceiver<CommittedSubDag>,
    ) {
        if let Ok(Some(_)) = timeout(Duration::from_secs(1), receiver.recv()).await {
            panic!("Expected the consensus output channel to be empty, but found more subdags.")
        }
    }
}
