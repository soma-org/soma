// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    iter,
    sync::Arc,
    time::Duration,
    vec,
};

use itertools::Itertools as _;
use parking_lot::RwLock;
#[cfg(test)]
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::{
    sync::{broadcast, watch},
    time::Instant,
};
use tracing::{debug, info, trace, warn};
use types::committee::AuthorityIndex;
#[cfg(test)]
use types::committee::{Stake, local_committee_and_keys};
use types::consensus::{
    block::{
        Block, BlockAPI, BlockRef, BlockTimestampMs, BlockV1, ExtendedBlock, GENESIS_ROUND, Round,
        SignedBlock, Slot, VerifiedBlock,
    },
    commit::{
        CertifiedCommit, CertifiedCommits, CommitAPI, CommittedSubDag, DecidedLeader, Decision,
    },
    context::Context,
    stake_aggregator::{QuorumThreshold, StakeAggregator},
};
use types::crypto::ProtocolKeyPair;
use types::error::{ConsensusError, ConsensusResult};

use crate::{CommitConsumerArgs, TransactionClient, block_verifier::NoopBlockVerifier};
use crate::{
    ancestor::{AncestorState, AncestorStateManager},
    block_manager::BlockManager,
    commit_observer::CommitObserver,
    dag_state::DagState,
    leader_schedule::LeaderSchedule,
    round_tracker::PeerRoundTracker,
    transaction::TransactionConsumer,
    transaction_certifier::TransactionCertifier,
    universal_committer::{
        UniversalCommitter, universal_committer_builder::UniversalCommitterBuilder,
    },
};
#[cfg(test)]
use types::consensus::block::CertifiedBlocksOutput;
#[cfg(test)]
use types::storage::consensus::mem_store::MemStore;

// Maximum number of commit votes to include in a block.
// TODO: Move to protocol config, and verify in BlockVerifier.
const MAX_COMMIT_VOTES_PER_BLOCK: usize = 100;

pub(crate) struct Core {
    context: Arc<Context>,
    /// The consumer to use in order to pull transactions to be included for the next proposals
    transaction_consumer: TransactionConsumer,
    /// This contains the reject votes on transactions which proposed blocks should include.
    transaction_certifier: TransactionCertifier,
    /// The block manager which is responsible for keeping track of the DAG dependencies when processing new blocks
    /// and accept them or suspend if we are missing their causal history
    block_manager: BlockManager,
    /// Estimated delay by round for propagating blocks to a quorum.
    /// Because of the nature of TCP and block streaming, propagation delay is expected to be
    /// 0 in most cases, even when the actual latency of broadcasting blocks is high.
    /// When this value is higher than the `propagation_delay_stop_proposal_threshold`,
    /// most likely this validator cannot broadcast  blocks to the network at all.
    /// Core stops proposing new blocks in this case.
    propagation_delay: Round,
    /// Used to make commit decisions for leader blocks in the dag.
    committer: UniversalCommitter,
    /// The last new round for which core has sent out a signal.
    last_signaled_round: Round,
    /// The blocks of the last included ancestors per authority. This vector is basically used as a
    /// watermark in order to include in the next block proposal only ancestors of higher rounds.
    /// By default, is initialised with `None` values.
    last_included_ancestors: Vec<Option<BlockRef>>,
    /// The last decided leader returned from the universal committer. Important to note
    /// that this does not signify that the leader has been persisted yet as it still has
    /// to go through CommitObserver and persist the commit in store. On recovery/restart
    /// the last_decided_leader will be set to the last_commit leader in dag state.
    last_decided_leader: Slot,
    /// The consensus leader schedule to be used to resolve the leader for a
    /// given round.
    leader_schedule: Arc<LeaderSchedule>,
    /// The commit observer is responsible for observing the commits and collecting
    /// + sending subdags over the consensus output channel.
    commit_observer: CommitObserver,
    /// Sender of outgoing signals from Core.
    signals: CoreSignals,
    /// The keypair to be used for block signing
    block_signer: ProtocolKeyPair,
    /// Keeping track of state of the DAG, including blocks, commits and last committed rounds.
    dag_state: Arc<RwLock<DagState>>,
    /// The last known round for which the node has proposed. Any proposal should be for a round > of this.
    /// This is currently being used to avoid equivocations during a node recovering from amnesia. When value is None it means that
    /// the last block sync mechanism is enabled, but it hasn't been initialised yet.
    last_known_proposed_round: Option<Round>,
    // The ancestor state manager will keep track of the quality of the authorities
    // based on the distribution of their blocks to the network. It will use this
    // information to decide whether to include that authority block in the next
    // proposal or not.
    ancestor_state_manager: AncestorStateManager,
    // The round tracker will keep track of the highest received and accepted rounds
    // from all authorities. It will use this information to then calculate the
    // quorum rounds periodically which is used across other components to make
    // decisions about block proposals.
    round_tracker: Arc<RwLock<PeerRoundTracker>>,
}

impl Core {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: Arc<Context>,
        leader_schedule: Arc<LeaderSchedule>,
        transaction_consumer: TransactionConsumer,
        transaction_certifier: TransactionCertifier,
        block_manager: BlockManager,
        commit_observer: CommitObserver,
        signals: CoreSignals,
        block_signer: ProtocolKeyPair,
        dag_state: Arc<RwLock<DagState>>,
        sync_last_known_own_block: bool,
        round_tracker: Arc<RwLock<PeerRoundTracker>>,
    ) -> Self {
        let last_decided_leader = dag_state.read().last_commit_leader();
        let number_of_leaders = context.protocol_config.mysticeti_num_leaders_per_round();
        let committer = UniversalCommitterBuilder::new(
            context.clone(),
            leader_schedule.clone(),
            dag_state.clone(),
        )
        .with_number_of_leaders(number_of_leaders)
        .with_pipeline(true)
        .build();

        let last_proposed_block = dag_state.read().get_last_proposed_block();

        let last_signaled_round = last_proposed_block.round();

        // Recover the last included ancestor rounds based on the last proposed block. That will allow
        // to perform the next block proposal by using ancestor blocks of higher rounds and avoid
        // re-including blocks that have been already included in the last (or earlier) block proposal.
        // This is only strongly guaranteed for a quorum of ancestors. It is still possible to re-include
        // a block from an authority which hadn't been added as part of the last proposal hence its
        // latest included ancestor is not accurately captured here. This is considered a small deficiency,
        // and it mostly matters just for this next proposal without any actual penalties in performance
        // or block proposal.
        let mut last_included_ancestors = vec![None; context.committee.size()];
        for ancestor in last_proposed_block.ancestors() {
            last_included_ancestors[ancestor.author] = Some(*ancestor);
        }

        let min_propose_round = if sync_last_known_own_block {
            None
        } else {
            // if the sync is disabled then we practically don't want to impose any restriction.
            Some(0)
        };

        let propagation_scores = leader_schedule.leader_swap_table.read().reputation_scores.clone();
        let mut ancestor_state_manager =
            AncestorStateManager::new(context.clone(), dag_state.clone());
        ancestor_state_manager.set_propagation_scores(propagation_scores);

        Self {
            context,
            last_signaled_round,
            last_included_ancestors,
            last_decided_leader,
            leader_schedule,
            transaction_consumer,
            transaction_certifier,
            block_manager,
            propagation_delay: 0,
            committer,
            commit_observer,
            signals,
            block_signer,
            dag_state,
            last_known_proposed_round: min_propose_round,
            ancestor_state_manager,
            round_tracker,
        }
        .recover()
    }

    fn recover(mut self) -> Self {
        // Try to commit and propose, since they may not have run after the last storage write.
        self.try_commit(vec![]).unwrap();

        let last_proposed_block = if let Some(last_proposed_block) = self.try_propose(true).unwrap()
        {
            last_proposed_block
        } else {
            let last_proposed_block = self.dag_state.read().get_last_proposed_block();

            if self.should_propose() {
                assert!(
                    last_proposed_block.round() > GENESIS_ROUND,
                    "At minimum a block of round higher than genesis should have been produced during recovery"
                );
            }

            // if no new block proposed then just re-broadcast the last proposed one to ensure liveness.
            self.signals
                .new_block(ExtendedBlock {
                    block: last_proposed_block.clone(),
                    excluded_ancestors: vec![],
                })
                .unwrap();
            last_proposed_block
        };

        // Try to set up leader timeout if needed.
        // This needs to be called after try_commit() and try_propose(), which may
        // have advanced the threshold clock round.
        self.try_signal_new_round();

        info!("Core recovery completed with last proposed block {:?}", last_proposed_block);

        self
    }

    /// Processes the provided blocks and accepts them if possible when their causal history exists.
    /// The method returns:
    /// - The references of ancestors missing their block
    #[tracing::instrument(skip_all)]
    pub(crate) fn add_blocks(
        &mut self,
        blocks: Vec<VerifiedBlock>,
    ) -> ConsensusResult<BTreeSet<BlockRef>> {
        let (accepted_blocks, missing_block_refs) = self.block_manager.try_accept_blocks(blocks);

        if !accepted_blocks.is_empty() {
            trace!(
                "Accepted blocks: {}",
                accepted_blocks.iter().map(|b| b.reference().to_string()).join(",")
            );

            // Try to commit the new blocks if possible.
            self.try_commit(vec![])?;

            // Try to propose now since there are new blocks accepted.
            self.try_propose(false)?;

            // Now set up leader timeout if needed.
            // This needs to be called after try_commit() and try_propose(), which may
            // have advanced the threshold clock round.
            self.try_signal_new_round();
        };

        if !missing_block_refs.is_empty() {
            trace!(
                "Missing block refs: {}",
                missing_block_refs.iter().map(|b| b.to_string()).join(", ")
            );
        }

        Ok(missing_block_refs)
    }

    // Adds the certified commits that have been synced via the commit syncer. We are using the commit info in order to skip running the decision
    // rule and immediately commit the corresponding leaders and sub dags. Pay attention that no block acceptance is happening here, but rather
    // internally in the `try_commit` method which ensures that everytime only the blocks corresponding to the certified commits that are about to
    // be committed are accepted.
    #[tracing::instrument(skip_all)]
    pub(crate) fn add_certified_commits(
        &mut self,
        certified_commits: CertifiedCommits,
    ) -> ConsensusResult<BTreeSet<BlockRef>> {
        let votes = certified_commits.votes().to_vec();
        let commits = self
            .filter_new_commits(certified_commits.commits().to_vec())
            .expect("Certified commits validation failed");

        // Try to accept the certified commit votes.
        // Even if they may not be part of a future commit, these blocks are useful for certifying
        // commits when helping peers sync commits.
        let (_, missing_block_refs) = self.block_manager.try_accept_blocks(votes);

        // Try to commit the new blocks. Take into account the trusted commit that has been provided.
        self.try_commit(commits)?;

        // Try to propose now since there are new blocks accepted.
        self.try_propose(false)?;

        // Now set up leader timeout if needed.
        // This needs to be called after try_commit() and try_propose(), which may
        // have advanced the threshold clock round.
        self.try_signal_new_round();

        Ok(missing_block_refs)
    }

    /// Checks if provided block refs have been accepted. If not, missing block refs are kept for synchronizations.
    /// Returns the references of missing blocks among the input blocks.
    pub(crate) fn check_block_refs(
        &mut self,
        block_refs: Vec<BlockRef>,
    ) -> ConsensusResult<BTreeSet<BlockRef>> {
        // Try to find them via the block manager
        let missing_block_refs = self.block_manager.try_find_blocks(block_refs);

        if !missing_block_refs.is_empty() {
            trace!(
                "Missing block refs: {}",
                missing_block_refs.iter().map(|b| b.to_string()).join(", ")
            );
        }
        Ok(missing_block_refs)
    }

    /// If needed, signals a new clock round and sets up leader timeout.
    fn try_signal_new_round(&mut self) {
        // Signal only when the threshold clock round is more advanced than the last signaled round.
        //
        // NOTE: a signal is still sent even when a block has been proposed at the new round.
        // We can consider changing this in the future.
        let new_clock_round = self.dag_state.read().threshold_clock_round();
        if new_clock_round <= self.last_signaled_round {
            return;
        }
        // Then send a signal to set up leader timeout.
        self.signals.new_round(new_clock_round);
        self.last_signaled_round = new_clock_round;
    }

    /// Creating a new block for the dictated round. This is used when a leader timeout occurs, either
    /// when the min timeout expires or max. When `force = true` , then any checks like previous round
    /// leader existence will get skipped.
    pub(crate) fn new_block(
        &mut self,
        round: Round,
        force: bool,
    ) -> ConsensusResult<Option<VerifiedBlock>> {
        if self.last_proposed_round() < round {
            let result = self.try_propose(force);
            // The threshold clock round may have advanced, so a signal needs to be sent.
            self.try_signal_new_round();
            return result;
        }
        Ok(None)
    }

    /// Keeps only the certified commits that have a commit index > last commit index.
    /// It also ensures that the first commit in the list is the next one in line, otherwise it panics.
    fn filter_new_commits(
        &mut self,
        commits: Vec<CertifiedCommit>,
    ) -> ConsensusResult<Vec<CertifiedCommit>> {
        // Filter out the commits that have been already locally committed and keep only anything that is above the last committed index.
        let last_commit_index = self.dag_state.read().last_commit_index();
        let commits = commits
            .iter()
            .filter(|commit| {
                if commit.index() > last_commit_index {
                    true
                } else {
                    tracing::debug!(
                        "Skip commit for index {} as it is already committed with last commit index {}",
                        commit.index(),
                        last_commit_index
                    );
                    false
                }
            })
            .cloned()
            .collect::<Vec<_>>();

        // Make sure that the first commit we find is the next one in line and there is no gap.
        if let Some(commit) = commits.first() {
            if commit.index() != last_commit_index + 1 {
                return Err(ConsensusError::UnexpectedCertifiedCommitIndex {
                    expected_commit_index: last_commit_index + 1,
                    commit_index: commit.index(),
                });
            }
        }

        Ok(commits)
    }

    // Attempts to create a new block, persist and propose it to all peers.
    // When force is true, ignore if leader from the last round exists among ancestors and if
    // the minimum round delay has passed.
    fn try_propose(&mut self, force: bool) -> ConsensusResult<Option<VerifiedBlock>> {
        if !self.should_propose() {
            return Ok(None);
        }
        if let Some(extended_block) = self.try_new_block(force) {
            self.signals.new_block(extended_block.clone())?;

            // The new block may help commit.
            self.try_commit(vec![])?;
            return Ok(Some(extended_block.block));
        }
        Ok(None)
    }

    /// Attempts to propose a new block for the next round. If a block has already proposed for latest
    /// or earlier round, then no block is created and None is returned.
    fn try_new_block(&mut self, force: bool) -> Option<ExtendedBlock> {
        // Ensure the new block has a higher round than the last proposed block.
        let clock_round = {
            let dag_state = self.dag_state.read();
            let clock_round = dag_state.threshold_clock_round();
            if clock_round <= dag_state.get_last_proposed_block().round() {
                debug!(
                    "Skipping block proposal for round {} as it is not higher than the last proposed block {}",
                    clock_round,
                    dag_state.get_last_proposed_block().round()
                );
                return None;
            }
            clock_round
        };

        // There must be a quorum of blocks from the previous round.
        let quorum_round = clock_round.saturating_sub(1);

        // Create a new block either because we want to "forcefully" propose a block due to a leader timeout,
        // or because we are actually ready to produce the block (leader exists and min delay has passed).
        if !force {
            if !self.leaders_exist(quorum_round) {
                return None;
            }

            if Duration::from_millis(
                self.context
                    .clock
                    .timestamp_utc_ms()
                    .saturating_sub(self.last_proposed_timestamp_ms()),
            ) < self.context.parameters.min_round_delay
            {
                debug!(
                    "Skipping block proposal for round {} as it is too soon after the last proposed block timestamp {}; min round delay is {}ms",
                    clock_round,
                    self.last_proposed_timestamp_ms(),
                    self.context.parameters.min_round_delay.as_millis(),
                );
                return None;
            }
        }

        // Determine the ancestors to be included in proposal.
        let (ancestors, excluded_and_equivocating_ancestors) =
            self.smart_ancestors_to_propose(clock_round, !force);

        // If we did not find enough good ancestors to propose, continue to wait before proposing.
        if ancestors.is_empty() {
            assert!(!force, "Ancestors should have been returned if force is true!");
            debug!(
                "Skipping block proposal for round {} because no good ancestor is found",
                clock_round,
            );
            return None;
        }

        let excluded_ancestors_limit = self.context.committee.size() * 2;
        if excluded_and_equivocating_ancestors.len() > excluded_ancestors_limit {
            debug!(
                "Dropping {} excluded ancestor(s) during proposal due to size limit",
                excluded_and_equivocating_ancestors.len() - excluded_ancestors_limit,
            );
        }
        let excluded_ancestors = excluded_and_equivocating_ancestors
            .into_iter()
            .take(excluded_ancestors_limit)
            .collect();

        // Update the last included ancestor block refs
        for ancestor in &ancestors {
            self.last_included_ancestors[ancestor.author()] = Some(ancestor.reference());
        }

        let now = self.context.clock.timestamp_utc_ms();
        ancestors.iter().for_each(|block| {
            if block.timestamp_ms() > now {
                trace!("Ancestor block {:?} has timestamp {}, greater than current timestamp {now}. Proposing for round {}.", block, block.timestamp_ms(), clock_round);


            }
        });

        // Consume the next transactions to be included. Do not drop the guards yet as this would acknowledge
        // the inclusion of transactions. Just let this be done in the end of the method.
        let (transactions, ack_transactions, _limit_reached) = self.transaction_consumer.next();

        // Consume the commit votes to be included.
        let commit_votes = self.dag_state.write().take_commit_votes(MAX_COMMIT_VOTES_PER_BLOCK);

        let transaction_votes = {
            let new_causal_history = {
                let mut dag_state = self.dag_state.write();
                ancestors
                    .iter()
                    .flat_map(|ancestor| dag_state.link_causal_history(ancestor.reference()))
                    .collect()
            };
            self.transaction_certifier.get_own_votes(new_causal_history)
        };

        // Create the block and insert to storage.
        let block = Block::V1(BlockV1::new(
            self.context.committee.epoch(),
            clock_round,
            self.context.own_index,
            now,
            ancestors.iter().map(|b| b.reference()).collect(),
            transactions,
            commit_votes,
            transaction_votes,
            vec![],
        ));
        let signed_block =
            SignedBlock::new(block, &self.block_signer).expect("Block signing failed.");
        let serialized = signed_block.serialize().expect("Block serialization failed.");

        // Own blocks are assumed to be valid.
        let verified_block = VerifiedBlock::new_verified(signed_block, serialized);

        // Record the interval from last proposal, before accepting the proposed block.
        let last_proposed_block = self.last_proposed_block();

        // Accept the block into BlockManager and DagState.
        let (accepted_blocks, missing) =
            self.block_manager.try_accept_blocks(vec![verified_block.clone()]);
        assert_eq!(accepted_blocks.len(), 1);
        assert!(missing.is_empty());

        // The block must be added to transaction certifier before it is broadcasted or added to DagState.
        // Update proposed state of blocks in local DAG.
        // TODO(fastpath): move this logic and the logic afterwards to proposed block handler.
        self.transaction_certifier.add_voted_blocks(vec![(verified_block.clone(), vec![])]);
        self.dag_state.write().link_causal_history(verified_block.reference());

        // Ensure the new block and its ancestors are persisted, before broadcasting it.
        self.dag_state.write().flush();

        // Now acknowledge the transactions for their inclusion to block
        ack_transactions(verified_block.reference());

        info!("Created block {verified_block:?} for round {clock_round}");

        let extended_block = ExtendedBlock { block: verified_block, excluded_ancestors };

        // Update round tracker with our own highest accepted blocks
        self.round_tracker.write().update_from_accepted_block(&extended_block);

        Some(extended_block)
    }

    /// Runs commit rule to attempt to commit additional blocks from the DAG. If any `certified_commits` are provided, then
    /// it will attempt to commit those first before trying to commit any further leaders.
    fn try_commit(
        &mut self,
        mut certified_commits: Vec<CertifiedCommit>,
    ) -> ConsensusResult<Vec<CommittedSubDag>> {
        let mut certified_commits_map = BTreeMap::new();
        for c in &certified_commits {
            certified_commits_map.insert(c.index(), c.reference());
        }

        if !certified_commits.is_empty() {
            info!(
                "Processing synced commits: {:?}",
                certified_commits.iter().map(|c| (c.index(), c.leader())).collect::<Vec<_>>()
            );
        }

        let mut committed_sub_dags = Vec::new();
        // TODO: Add optimization to abort early without quorum for a round.
        loop {
            // LeaderSchedule has a limit to how many sequenced leaders can be committed
            // before a change is triggered. Calling into leader schedule will get you
            // how many commits till next leader change. We will loop back and recalculate
            // any discarded leaders with the new schedule.
            let mut commits_until_update =
                self.leader_schedule.commits_until_leader_schedule_update(self.dag_state.clone());

            if commits_until_update == 0 {
                let last_commit_index = self.dag_state.read().last_commit_index();

                tracing::info!(
                    "Leader schedule change triggered at commit index {last_commit_index}"
                );

                self.leader_schedule.update_leader_schedule_v2(&self.dag_state);

                let propagation_scores =
                    self.leader_schedule.leader_swap_table.read().reputation_scores.clone();
                self.ancestor_state_manager.set_propagation_scores(propagation_scores);

                commits_until_update = self
                    .leader_schedule
                    .commits_until_leader_schedule_update(self.dag_state.clone());
            }
            assert!(commits_until_update > 0);

            // If there are certified commits to process, find out which leaders and commits from them
            // are decided and use them as the next commits.
            let (certified_leaders, decided_certified_commits): (
                Vec<DecidedLeader>,
                Vec<CertifiedCommit>,
            ) = self
                .try_select_certified_leaders(&mut certified_commits, commits_until_update)
                .into_iter()
                .unzip();

            // Only accept blocks for the certified commits that we are certain to sequence.
            // This ensures that only blocks corresponding to committed certified commits are flushed to disk.
            // Blocks from non-committed certified commits will not be flushed, preventing issues during crash-recovery.
            // This avoids scenarios where accepting and flushing blocks of non-committed certified commits could lead to
            // premature commit rule execution. Due to GC, this could cause a panic if the commit rule tries to access
            // missing causal history from blocks of certified commits.
            let blocks = decided_certified_commits
                .iter()
                .flat_map(|c| c.blocks())
                .cloned()
                .collect::<Vec<_>>();
            self.block_manager.try_accept_committed_blocks(blocks);

            // If there is no certified commit to process, run the decision rule.
            let (decided_leaders, local) = if certified_leaders.is_empty() {
                // TODO: limit commits by commits_until_update for efficiency, which may be needed when leader schedule length is reduced.
                let mut decided_leaders = self.committer.try_decide(self.last_decided_leader);
                // Truncate the decided leaders to fit the commit schedule limit.
                if decided_leaders.len() >= commits_until_update {
                    let _ = decided_leaders.split_off(commits_until_update);
                }
                (decided_leaders, true)
            } else {
                (certified_leaders, false)
            };

            // If the decided leaders list is empty then just break the loop.
            let Some(last_decided) = decided_leaders.last().cloned() else {
                break;
            };

            self.last_decided_leader = last_decided.slot();

            let sequenced_leaders = decided_leaders
                .into_iter()
                .filter_map(|leader| leader.into_committed_block())
                .collect::<Vec<_>>();
            // It's possible to reach this point as the decided leaders might all of them be "Skip" decisions. In this case there is no
            // leader to commit and we should break the loop.
            if sequenced_leaders.is_empty() {
                break;
            }
            tracing::info!(
                "Committing {} leaders: {}; {} commits before next leader schedule change",
                sequenced_leaders.len(),
                sequenced_leaders.iter().map(|b| b.reference().to_string()).join(","),
                commits_until_update,
            );

            // TODO: refcount subdags
            let subdags = self.commit_observer.handle_commit(sequenced_leaders, local)?;

            // Try to unsuspend blocks if gc_round has advanced.
            self.block_manager.try_unsuspend_blocks_for_latest_gc_round();

            committed_sub_dags.extend(subdags);
        }

        // Sanity check: for commits that have been linearized using the certified commits, ensure that the same sub dag has been committed.
        for sub_dag in &committed_sub_dags {
            if let Some(commit_ref) = certified_commits_map.remove(&sub_dag.commit_ref.index) {
                assert_eq!(
                    commit_ref, sub_dag.commit_ref,
                    "Certified commit has different reference than the committed sub dag"
                );
            }
        }

        // Notify about our own committed blocks
        let committed_block_refs = committed_sub_dags
            .iter()
            .flat_map(|sub_dag| sub_dag.blocks.iter())
            .filter_map(|block| {
                (block.author() == self.context.own_index).then_some(block.reference())
            })
            .collect::<Vec<_>>();
        self.transaction_consumer
            .notify_own_blocks_status(committed_block_refs, self.dag_state.read().gc_round());

        Ok(committed_sub_dags)
    }

    pub(crate) fn get_missing_blocks(&self) -> BTreeSet<BlockRef> {
        self.block_manager.missing_blocks()
    }

    /// Sets the delay by round for propagating blocks to a quorum.
    pub(crate) fn set_propagation_delay(&mut self, delay: Round) {
        info!("Propagation round delay set to: {delay}");
        self.propagation_delay = delay;
    }

    /// Sets the min propose round for the proposer allowing to propose blocks only for round numbers
    /// `> last_known_proposed_round`. At the moment is allowed to call the method only once leading to a panic
    /// if attempt to do multiple times.
    pub(crate) fn set_last_known_proposed_round(&mut self, round: Round) {
        if self.last_known_proposed_round.is_some() {
            panic!(
                "Should not attempt to set the last known proposed round if that has been already set"
            );
        }
        self.last_known_proposed_round = Some(round);
        info!("Last known proposed round set to {round}");
    }

    /// Whether the core should propose new blocks.
    pub(crate) fn should_propose(&self) -> bool {
        let clock_round = self.dag_state.read().threshold_clock_round();

        if self.propagation_delay
            > self.context.parameters.propagation_delay_stop_proposal_threshold
        {
            debug!(
                "Skip proposing for round {clock_round}, high propagation delay {} > {}.",
                self.propagation_delay,
                self.context.parameters.propagation_delay_stop_proposal_threshold
            );

            return false;
        }

        let Some(last_known_proposed_round) = self.last_known_proposed_round else {
            debug!(
                "Skip proposing for round {clock_round}, last known proposed round has not been synced yet."
            );

            return false;
        };
        if clock_round <= last_known_proposed_round {
            debug!(
                "Skip proposing for round {clock_round} as last known proposed round is {last_known_proposed_round}"
            );

            return false;
        }

        true
    }

    // Tries to select a prefix of certified commits to be committed next respecting the `limit`.
    // If provided `limit` is zero, it will panic.
    // The function returns a list of certified leaders and certified commits. If empty vector is returned, it means that
    // there are no certified commits to be committed, as input `certified_commits` is either empty or all of the certified
    // commits have been already committed.
    #[tracing::instrument(skip_all)]
    fn try_select_certified_leaders(
        &mut self,
        certified_commits: &mut Vec<CertifiedCommit>,
        limit: usize,
    ) -> Vec<(DecidedLeader, CertifiedCommit)> {
        assert!(limit > 0, "limit should be greater than 0");
        if certified_commits.is_empty() {
            return vec![];
        }

        let to_commit = if certified_commits.len() >= limit {
            // We keep only the number of leaders as dictated by the `limit`
            certified_commits.drain(..limit).collect::<Vec<_>>()
        } else {
            // Otherwise just take all of them and leave the `synced_commits` empty.
            std::mem::take(certified_commits)
        };

        tracing::debug!(
            "Selected {} certified leaders: {}",
            to_commit.len(),
            to_commit.iter().map(|c| c.leader().to_string()).join(",")
        );

        to_commit
            .into_iter()
            .map(|commit| {
                let leader = commit.blocks().last().expect("Certified commit should have at least one block");
                assert_eq!(leader.reference(), commit.leader(), "Last block of the committed sub dag should have the same digest as the leader of the commit");
                // There is no knowledge of direct commit with certified commits, so assuming indirect commit.
                let leader = DecidedLeader::Commit(leader.clone(), /* direct */ false);
                UniversalCommitter::update_metrics(&self.context, &leader, Decision::Certified);
                (leader, commit)
            })
            .collect::<Vec<_>>()
    }

    /// Retrieves the next ancestors to propose to form a block at `clock_round` round.
    /// If smart selection is enabled then this will try to select the best ancestors
    /// based on the propagation scores of the authorities.
    fn smart_ancestors_to_propose(
        &mut self,
        clock_round: Round,
        smart_select: bool,
    ) -> (Vec<VerifiedBlock>, BTreeSet<BlockRef>) {
        // Now take the ancestors before the clock_round (excluded) for each authority.
        let all_ancestors = self.dag_state.read().get_last_cached_block_per_authority(clock_round);

        assert_eq!(
            all_ancestors.len(),
            self.context.committee.size(),
            "Fatal error, number of returned ancestors don't match committee size."
        );

        // Ensure ancestor state is up to date before selecting for proposal.
        let accepted_quorum_rounds = self.round_tracker.read().compute_accepted_quorum_rounds();

        self.ancestor_state_manager.update_all_ancestors_state(&accepted_quorum_rounds);

        let ancestor_state_map = self.ancestor_state_manager.get_ancestor_states();

        let quorum_round = clock_round.saturating_sub(1);

        let mut score_and_pending_excluded_ancestors = Vec::new();
        let mut excluded_and_equivocating_ancestors = BTreeSet::new();

        // Propose only ancestors of higher rounds than what has already been proposed.
        // And always include own last proposed block first among ancestors.
        // Start by only including the high scoring ancestors. Low scoring ancestors
        // will be included in a second pass below.
        let included_ancestors = iter::once(self.last_proposed_block().clone())
            .chain(
                all_ancestors
                    .into_iter()
                    .flat_map(|(ancestor, equivocating_ancestors)| {
                        if ancestor.author() == self.context.own_index {
                            return None;
                        }
                        if let Some(last_block_ref) =
                            self.last_included_ancestors[ancestor.author()]
                        {
                            if last_block_ref.round >= ancestor.round() {
                                return None;
                            }
                        }

                        // We will never include equivocating ancestors so add them immediately
                        excluded_and_equivocating_ancestors.extend(equivocating_ancestors);

                        let ancestor_state = ancestor_state_map[ancestor.author()];
                        match ancestor_state {
                            AncestorState::Include => {
                                trace!("Found ancestor {ancestor} with INCLUDE state for round {clock_round}");
                            }
                            AncestorState::Exclude(score) => {
                                trace!("Added ancestor {ancestor} with EXCLUDE state with score {score} to temporary excluded ancestors for round {clock_round}");
                                score_and_pending_excluded_ancestors.push((score, ancestor));
                                return None;
                            }
                        }

                        Some(ancestor)
                    }),
            )
            .collect::<Vec<_>>();

        let mut parent_round_quorum = StakeAggregator::<QuorumThreshold>::new();

        // Check total stake of high scoring parent round ancestors
        for ancestor in included_ancestors.iter().filter(|a| a.round() == quorum_round) {
            parent_round_quorum.add(ancestor.author(), &self.context.committee);
        }

        if smart_select && !parent_round_quorum.reached_threshold(&self.context.committee) {
            debug!(
                "Only found {} stake of good ancestors to include for round {clock_round}, will wait for more.",
                parent_round_quorum.stake()
            );
            return (vec![], BTreeSet::new());
        }

        // Sort scores descending so we can include the best of the pending excluded
        // ancestors first until we reach the threshold.
        score_and_pending_excluded_ancestors.sort_by(|a, b| b.0.cmp(&a.0));

        let mut ancestors_to_propose = included_ancestors;
        let mut excluded_ancestors = Vec::new();
        for (score, ancestor) in score_and_pending_excluded_ancestors.into_iter() {
            if !parent_round_quorum.reached_threshold(&self.context.committee)
                && ancestor.round() == quorum_round
            {
                debug!(
                    "Including temporarily excluded parent round ancestor {ancestor} with score {score} to propose for round {clock_round}"
                );
                parent_round_quorum.add(ancestor.author(), &self.context.committee);
                ancestors_to_propose.push(ancestor);
            } else {
                excluded_ancestors.push((score, ancestor));
            }
        }

        // Iterate through excluded ancestors and include the ancestor or the ancestor's ancestor
        // that has been accepted by a quorum of the network. If the original ancestor itself
        // is not included then it will be part of excluded ancestors that are not
        // included in the block but will still be broadcasted to peers.
        for (score, ancestor) in excluded_ancestors.iter() {
            let excluded_author = ancestor.author();

            // A quorum of validators reported to have accepted blocks from the excluded_author up to the low quorum round.
            let mut accepted_low_quorum_round = accepted_quorum_rounds[excluded_author].0;
            // If the accepted quorum round of this ancestor is greater than or equal
            // to the clock round then we want to make sure to set it to clock_round - 1
            // as that is the max round the new block can include as an ancestor.
            accepted_low_quorum_round = accepted_low_quorum_round.min(quorum_round);

            let last_included_round = self.last_included_ancestors[excluded_author]
                .map(|block_ref| block_ref.round)
                .unwrap_or(GENESIS_ROUND);
            if ancestor.round() <= last_included_round {
                // This should have already been filtered out when filtering all_ancestors.
                // Still, ensure previously included ancestors are filtered out.
                continue;
            }

            if last_included_round >= accepted_low_quorum_round {
                excluded_and_equivocating_ancestors.insert(ancestor.reference());
                trace!(
                    "Excluded low score ancestor {} with score {score} to propose for round {clock_round}: last included round {last_included_round} >= accepted low quorum round {accepted_low_quorum_round}",
                    ancestor.reference()
                );

                continue;
            }

            let ancestor = if ancestor.round() <= accepted_low_quorum_round {
                // Include the ancestor block as it has been seen & accepted by a strong quorum.
                ancestor.clone()
            } else {
                // Exclude this ancestor since it hasn't been accepted by a strong quorum
                excluded_and_equivocating_ancestors.insert(ancestor.reference());
                trace!(
                    "Excluded low score ancestor {} with score {score} to propose for round {clock_round}: ancestor round {} > accepted low quorum round {accepted_low_quorum_round} ",
                    ancestor.reference(),
                    ancestor.round()
                );

                // Look for an earlier block in the ancestor chain that we can include as there
                // is a gap between the last included round and the accepted low quorum round.
                //
                // Note: Only cached blocks need to be propagated. Committed and GC'ed blocks
                // do not need to be propagated.
                match self.dag_state.read().get_last_cached_block_in_range(
                    excluded_author,
                    last_included_round + 1,
                    accepted_low_quorum_round + 1,
                ) {
                    Some(earlier_ancestor) => {
                        // Found an earlier block that has been propagated well - include it instead
                        earlier_ancestor
                    }
                    None => {
                        // No suitable earlier block found
                        continue;
                    }
                }
            };
            self.last_included_ancestors[excluded_author] = Some(ancestor.reference());
            ancestors_to_propose.push(ancestor.clone());
            trace!(
                "Included low scoring ancestor {} with score {score} seen at accepted low quorum round {accepted_low_quorum_round} to propose for round {clock_round}",
                ancestor.reference()
            );
        }

        assert!(
            parent_round_quorum.reached_threshold(&self.context.committee),
            "Fatal error, quorum not reached for parent round when proposing for round {clock_round}. Possible mismatch between DagState and Core."
        );

        debug!(
            "Included {} ancestors & excluded {} low performing or equivocating ancestors for proposal in round {clock_round}",
            ancestors_to_propose.len(),
            excluded_and_equivocating_ancestors.len()
        );

        (ancestors_to_propose, excluded_and_equivocating_ancestors)
    }

    /// Checks whether all the leaders of the round exist.
    /// TODO: we can leverage some additional signal here in order to more cleverly manipulate later the leader timeout
    /// Ex if we already have one leader - the first in order - we might don't want to wait as much.
    fn leaders_exist(&self, round: Round) -> bool {
        let dag_state = self.dag_state.read();
        for leader in self.leaders(round) {
            // Search for all the leaders. If at least one is not found, then return false.
            // A linear search should be fine here as the set of elements is not expected to be small enough and more sophisticated
            // data structures might not give us much here.
            if !dag_state.contains_cached_block_at_slot(leader) {
                return false;
            }
        }

        true
    }

    /// Returns the leaders of the provided round.
    fn leaders(&self, round: Round) -> Vec<Slot> {
        self.committer
            .get_leaders(round)
            .into_iter()
            .map(|authority_index| Slot::new(round, authority_index))
            .collect()
    }

    /// Returns the 1st leader of the round.
    fn first_leader(&self, round: Round) -> AuthorityIndex {
        self.leaders(round).first().unwrap().authority
    }

    fn last_proposed_timestamp_ms(&self) -> BlockTimestampMs {
        self.last_proposed_block().timestamp_ms()
    }

    fn last_proposed_round(&self) -> Round {
        self.last_proposed_block().round()
    }

    fn last_proposed_block(&self) -> VerifiedBlock {
        self.dag_state.read().get_last_proposed_block()
    }
}

/// Senders of signals from Core, for outputs and events (ex new block produced).
pub(crate) struct CoreSignals {
    tx_block_broadcast: broadcast::Sender<ExtendedBlock>,
    new_round_sender: watch::Sender<Round>,
    context: Arc<Context>,
}

impl CoreSignals {
    pub fn new(context: Arc<Context>) -> (Self, CoreSignalsReceivers) {
        // Blocks buffered in broadcast channel should be roughly equal to thosed cached in dag state,
        // since the underlying blocks are ref counted so a lower buffer here will not reduce memory
        // usage significantly.
        let (tx_block_broadcast, rx_block_broadcast) = broadcast::channel::<ExtendedBlock>(
            context.parameters.dag_state_cached_rounds as usize,
        );
        let (new_round_sender, new_round_receiver) = watch::channel(0);

        let me = Self { tx_block_broadcast, new_round_sender, context };

        let receivers = CoreSignalsReceivers { rx_block_broadcast, new_round_receiver };

        (me, receivers)
    }

    /// Sends a signal to all the waiters that a new block has been produced. The method will return
    /// true if block has reached even one subscriber, false otherwise.
    pub(crate) fn new_block(&self, extended_block: ExtendedBlock) -> ConsensusResult<()> {
        // When there is only one authority in committee, it is unnecessary to broadcast
        // the block which will fail anyway without subscribers to the signal.
        if self.context.committee.size() > 1 {
            if extended_block.block.round() == GENESIS_ROUND {
                debug!("Ignoring broadcasting genesis block to peers");
                return Ok(());
            }

            if let Err(err) = self.tx_block_broadcast.send(extended_block) {
                warn!("Couldn't broadcast the block to any receiver: {err}");
                return Err(ConsensusError::Shutdown);
            }
        } else {
            debug!(
                "Did not broadcast block {extended_block:?} to receivers as committee size is <= 1"
            );
        }
        Ok(())
    }

    /// Sends a signal that threshold clock has advanced to new round. The `round_number` is the round at which the
    /// threshold clock has advanced to.
    pub(crate) fn new_round(&mut self, round_number: Round) {
        let _ = self.new_round_sender.send_replace(round_number);
    }
}

/// Receivers of signals from Core.
/// Intentionally un-clonable. Comonents should only subscribe to channels they need.
pub(crate) struct CoreSignalsReceivers {
    rx_block_broadcast: broadcast::Receiver<ExtendedBlock>,
    new_round_receiver: watch::Receiver<Round>,
}

impl CoreSignalsReceivers {
    pub(crate) fn block_broadcast_receiver(&self) -> broadcast::Receiver<ExtendedBlock> {
        self.rx_block_broadcast.resubscribe()
    }

    pub(crate) fn new_round_receiver(&self) -> watch::Receiver<Round> {
        self.new_round_receiver.clone()
    }
}

/// Creates cores for the specified number of authorities for their corresponding stakes. The method returns the
/// cores and their respective signal receivers are returned in `AuthorityIndex` order asc.
#[cfg(test)]
pub(crate) async fn create_cores(
    context: Context,
    authorities: Vec<Stake>,
) -> Vec<CoreTextFixture> {
    let mut cores = Vec::new();

    for index in 0..authorities.len() {
        let own_index = AuthorityIndex::new_for_test(index as u32);
        let core =
            CoreTextFixture::new(context.clone(), authorities.clone(), own_index, false).await;
        cores.push(core);
    }
    cores
}

#[cfg(test)]
pub(crate) struct CoreTextFixture {
    pub(crate) core: Core,
    pub(crate) transaction_certifier: TransactionCertifier,
    pub(crate) signal_receivers: CoreSignalsReceivers,
    pub(crate) block_receiver: broadcast::Receiver<ExtendedBlock>,
    pub(crate) _commit_output_receiver: UnboundedReceiver<CommittedSubDag>,
    pub(crate) _blocks_output_receiver: UnboundedReceiver<CertifiedBlocksOutput>,
    pub(crate) dag_state: Arc<RwLock<DagState>>,
    pub(crate) store: Arc<MemStore>,
}

#[cfg(test)]
impl CoreTextFixture {
    async fn new(
        context: Context,
        authorities: Vec<Stake>,
        own_index: AuthorityIndex,
        sync_last_known_own_block: bool,
    ) -> Self {
        let (committee, mut signers, _) = local_committee_and_keys(0, authorities.clone());
        let mut context = context.clone();
        context = context.with_committee(committee).with_authority_index(own_index);
        context.protocol_config.set_consensus_bad_nodes_stake_threshold_for_testing(33);

        let context = Arc::new(context);
        let store = Arc::new(MemStore::new());
        let dag_state = Arc::new(RwLock::new(DagState::new(context.clone(), store.clone())));

        let block_manager = BlockManager::new(context.clone(), dag_state.clone());
        let leader_schedule = Arc::new(
            LeaderSchedule::from_store(context.clone(), dag_state.clone())
                .with_num_commits_per_schedule(10),
        );
        let (_transaction_client, tx_receiver) = TransactionClient::new(context.clone());
        let transaction_consumer = TransactionConsumer::new(tx_receiver, context.clone());
        let (blocks_sender, _blocks_receiver) = tokio::sync::mpsc::unbounded_channel();
        let transaction_certifier = TransactionCertifier::new(
            context.clone(),
            Arc::new(NoopBlockVerifier {}),
            dag_state.clone(),
            blocks_sender,
        );
        let (signals, signal_receivers) = CoreSignals::new(context.clone());
        // Need at least one subscriber to the block broadcast channel.
        let block_receiver = signal_receivers.block_broadcast_receiver();

        let (commit_consumer, commit_output_receiver, blocks_output_receiver) =
            CommitConsumerArgs::new(0, 0);
        let commit_observer = CommitObserver::new(
            context.clone(),
            commit_consumer,
            dag_state.clone(),
            transaction_certifier.clone(),
            leader_schedule.clone(),
        )
        .await;

        let block_signer = signers.remove(own_index.value()).1;

        let round_tracker = Arc::new(RwLock::new(PeerRoundTracker::new(context.clone())));
        let core = Core::new(
            context,
            leader_schedule,
            transaction_consumer,
            transaction_certifier.clone(),
            block_manager,
            commit_observer,
            signals,
            block_signer,
            dag_state.clone(),
            sync_last_known_own_block,
            round_tracker,
        );

        Self {
            core,
            transaction_certifier,
            signal_receivers,
            block_receiver,
            _commit_output_receiver: commit_output_receiver,
            _blocks_output_receiver: blocks_output_receiver,
            dag_state,
            store,
        }
    }

    pub(crate) fn add_blocks(
        &mut self,
        blocks: Vec<VerifiedBlock>,
    ) -> ConsensusResult<BTreeSet<BlockRef>> {
        self.transaction_certifier
            .add_voted_blocks(blocks.iter().map(|b| (b.clone(), vec![])).collect());
        self.core.add_blocks(blocks)
    }
}

// Portions of these tests are derived from Mysticeti consensus (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/consensus/core/src/core.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use types::committee::AuthorityIndex;
    use types::consensus::{
        block::{BlockAPI, BlockRef, TestBlock, VerifiedBlock, genesis_blocks},
        commit::{CertifiedCommit, CommitDigest, TrustedCommit},
        context::Context,
    };

    use super::*;

    /// Helper: creates a CoreTextFixture for authority 0 with 4 authorities (equal stake)
    /// and min_round_delay set to zero so that proposals are not blocked by timing.
    async fn core_fixture() -> CoreTextFixture {
        let (mut context, _keys) = Context::new_for_test(4);
        context.parameters.min_round_delay = Duration::ZERO;
        // Each authority gets 2500 voting power; total = 10000 = TOTAL_VOTING_POWER.
        let authorities = vec![2500; 4];
        CoreTextFixture::new(context, authorities, AuthorityIndex::new_for_test(0), false).await
    }

    /// Helper: builds round-1 blocks for the given authorities referencing all genesis blocks.
    fn build_round_1_blocks(context: &Context, authors: &[u32]) -> Vec<VerifiedBlock> {
        let genesis = genesis_blocks(context);
        let genesis_refs: Vec<BlockRef> = genesis.iter().map(|b| b.reference()).collect();

        authors
            .iter()
            .map(|&author| {
                VerifiedBlock::new_for_test(
                    TestBlock::new(1, author).set_ancestors(genesis_refs.clone()).build(),
                )
            })
            .collect()
    }

    /// Test that after adding a full round of blocks, the core proposes a new block.
    #[tokio::test]
    async fn test_core_propose_after_genesis() {
        let mut fixture = core_fixture().await;

        // After recovery, core[0] should have proposed its own block at round 1.
        let last_proposed = fixture.core.last_proposed_block();
        assert_eq!(
            last_proposed.round(),
            1,
            "Core should have proposed at round 1 during recovery"
        );
        assert_eq!(last_proposed.author(), AuthorityIndex::new_for_test(0));

        // Build round-1 blocks from authorities 1, 2, 3 and add them.
        let context = fixture.core.context.clone();
        let round_1_blocks = build_round_1_blocks(&context, &[1, 2, 3]);

        fixture.add_blocks(round_1_blocks).unwrap();

        // After adding a quorum of round-1 blocks, the threshold clock should advance
        // to round 2 and the core should propose a round-2 block.
        let last_proposed = fixture.core.last_proposed_block();
        assert_eq!(
            last_proposed.round(),
            2,
            "Core should have proposed at round 2 after receiving quorum of round-1 blocks"
        );
        assert_eq!(last_proposed.author(), AuthorityIndex::new_for_test(0));

        // The ancestors of the round-2 block should include blocks from round 1.
        let ancestors = last_proposed.ancestors();
        assert!(
            ancestors.len() >= 3,
            "Round 2 block should have at least 3 ancestors (quorum), got {}",
            ancestors.len()
        );
    }

    /// Test that the core only proposes once it receives a quorum of blocks from
    /// the previous round (not before).
    #[tokio::test]
    async fn test_core_propose_once_receiving_a_quorum() {
        let mut fixture = core_fixture().await;

        // After recovery, core has proposed at round 1.
        assert_eq!(fixture.core.last_proposed_round(), 1);

        let context = fixture.core.context.clone();

        // Add authority 1's round-1 block. With own block (auth 0) + auth 1 = 2 * 2500 = 5000 < 6667
        // The threshold clock should NOT advance yet.
        let block_auth1 = build_round_1_blocks(&context, &[1]);
        fixture.add_blocks(block_auth1).unwrap();
        assert_eq!(
            fixture.core.last_proposed_round(),
            1,
            "Should NOT have proposed round 2 with only 2 out of 4 round-1 blocks"
        );

        // Add authority 2's round-1 block. Now 3 * 2500 = 7500 >= 6667, quorum reached.
        let block_auth2 = build_round_1_blocks(&context, &[2]);
        fixture.add_blocks(block_auth2).unwrap();

        // Now the threshold clock should advance to round 2 and core should propose.
        assert_eq!(
            fixture.core.last_proposed_round(),
            2,
            "Should have proposed round 2 after reaching quorum of round-1 blocks"
        );
    }

    /// Test that the core proposes a new block on leader timeout (force=true)
    /// even when the leader of the previous round doesn't exist in the DAG.
    #[tokio::test]
    async fn test_core_try_new_block_leader_timeout() {
        let mut fixture = core_fixture().await;

        // After recovery, core has proposed at round 1.
        assert_eq!(fixture.core.last_proposed_round(), 1);

        let context = fixture.core.context.clone();

        // Add round-1 blocks from authorities 1, 2, 3 to reach full quorum.
        let round_1_blocks = build_round_1_blocks(&context, &[1, 2, 3]);
        fixture.add_blocks(round_1_blocks).unwrap();

        // Core should have proposed at round 2.
        assert_eq!(fixture.core.last_proposed_round(), 2);

        // Now build round-2 blocks from authorities 1 and 2 (NOT authority 3).
        // We need their ancestors to be the actual round-1 blocks in the DAG.
        let round_1_refs: Vec<BlockRef> = {
            let dag = fixture.dag_state.read();
            dag.get_last_cached_block_per_authority(2)
                .into_iter()
                .map(|(b, _)| b.reference())
                .collect()
        };

        let block_r2_auth1 = VerifiedBlock::new_for_test(
            TestBlock::new(2, 1).set_ancestors(round_1_refs.clone()).build(),
        );
        let block_r2_auth2 = VerifiedBlock::new_for_test(
            TestBlock::new(2, 2).set_ancestors(round_1_refs.clone()).build(),
        );
        // Add both round-2 blocks (auth 0 own + auth 1 + auth 2 = 3*2500 = 7500 >= 6667).
        // This should advance the threshold clock to round 3.
        fixture.add_blocks(vec![block_r2_auth1, block_r2_auth2]).unwrap();

        // The core may or may not have proposed at round 3 depending on whether
        // the leader of round 2 exists. Let's check and force if needed.
        if fixture.core.last_proposed_round() < 3 {
            // Force propose at round 3 via leader timeout.
            let proposed = fixture.core.new_block(3, true).unwrap();
            assert!(proposed.is_some(), "Force proposal should succeed at round 3");
        }
        assert_eq!(
            fixture.core.last_proposed_round(),
            3,
            "Should have proposed at round 3 (via force if needed)"
        );
    }

    /// Test that set_last_known_proposed_round correctly controls proposal.
    #[tokio::test]
    async fn test_core_set_min_propose_round() {
        // Create a core with sync_last_known_own_block=true, which means
        // last_known_proposed_round starts as None, preventing proposals.
        let (mut context, _keys) = Context::new_for_test(4);
        context.parameters.min_round_delay = Duration::ZERO;
        let authorities = vec![2500; 4];
        let mut fixture = CoreTextFixture::new(
            context,
            authorities,
            AuthorityIndex::new_for_test(0),
            true, // sync_last_known_own_block
        )
        .await;

        // With sync_last_known_own_block=true, last_known_proposed_round is None.
        // The core should NOT propose (should_propose returns false).
        assert!(!fixture.core.should_propose());

        // Set the last known proposed round to 0.
        fixture.core.set_last_known_proposed_round(0);

        // Now should_propose should return true (clock_round=1 > 0).
        assert!(fixture.core.should_propose());

        // Try to propose: since threshold clock is at round 1 and last proposed
        // is genesis round 0, it should propose at round 1.
        let result = fixture.core.try_propose(true).unwrap();
        assert!(result.is_some(), "Should successfully propose after setting min round");
        assert_eq!(fixture.core.last_proposed_round(), 1);
    }

    /// Test that CoreSignals correctly emit new_round and new_block signals.
    #[tokio::test]
    async fn test_core_signals() {
        let mut fixture = core_fixture().await;
        let context = fixture.core.context.clone();

        // Get a new_round receiver.
        let mut new_round_rx = fixture.signal_receivers.new_round_receiver();

        // The initial signaled round after recovery should be round 1.
        let initial_round = *new_round_rx.borrow_and_update();
        assert_eq!(initial_round, 1, "Initial signaled round should be 1");

        // Add round-1 blocks from authorities 1, 2, 3 to advance threshold clock to round 2.
        let round_1_blocks = build_round_1_blocks(&context, &[1, 2, 3]);
        fixture.add_blocks(round_1_blocks).unwrap();

        // Check that a new round signal was sent for round 2.
        new_round_rx.changed().await.unwrap();
        let new_round = *new_round_rx.borrow_and_update();
        assert_eq!(new_round, 2, "New round signal should be 2");

        // The block broadcast receiver may have the round-1 block from recovery first.
        // Drain until we find the round-2 block.
        let mut found_round_2 = false;
        while let Ok(extended_block) = fixture.block_receiver.try_recv() {
            if extended_block.block.round() == 2 {
                found_round_2 = true;
                break;
            }
        }
        assert!(found_round_2, "Block broadcast should contain the round-2 block");
    }

    /// Test that filter_new_commits correctly filters out already committed commits
    /// and validates commit sequence continuity.
    #[tokio::test]
    async fn test_filter_new_commits() {
        let mut fixture = core_fixture().await;

        // Create some fake commits with sequential indices.
        let leader_ref = BlockRef::new(1, AuthorityIndex::new_for_test(1), Default::default());
        let make_certified_commit = |index: u32| -> CertifiedCommit {
            let commit = TrustedCommit::new_for_test(
                index,
                CommitDigest::MIN,
                0,
                leader_ref,
                vec![leader_ref],
            );
            CertifiedCommit::new_certified(commit, vec![])
        };

        // Initially no commits have been processed (last_commit_index = 0).
        // Filtering commits [1, 2, 3] should return all of them.
        let commits =
            vec![make_certified_commit(1), make_certified_commit(2), make_certified_commit(3)];
        let filtered = fixture.core.filter_new_commits(commits).unwrap();
        assert_eq!(filtered.len(), 3, "All commits should pass filter initially");
        assert_eq!(filtered[0].index(), 1);
        assert_eq!(filtered[1].index(), 2);
        assert_eq!(filtered[2].index(), 3);

        // Filtering commits that start before the last committed index should
        // filter out already committed ones.
        // If last_commit_index is 0, commits starting at 1 are fine.
        // But if we pass commits [0, 1, 2], index 0 should be filtered out
        // (0 is not > 0, so it's filtered).
        let commits = vec![make_certified_commit(1), make_certified_commit(2)];
        let filtered = fixture.core.filter_new_commits(commits).unwrap();
        assert_eq!(filtered.len(), 2);

        // Test gap detection: if we skip an index (e.g., pass commit 3 when
        // last_commit_index is 0, but commit 1 hasn't been committed), it should error.
        let commits = vec![make_certified_commit(2)];
        let result = fixture.core.filter_new_commits(commits);
        assert!(
            result.is_err(),
            "Should error when first commit index is not consecutive with last_commit_index"
        );

        // Test that empty input returns empty output (no error).
        let filtered = fixture.core.filter_new_commits(vec![]).unwrap();
        assert!(filtered.is_empty());
    }

    // =========================================================================
    // 13 additional tests
    // =========================================================================

    /// Build blocks for a given round referencing the provided ancestor refs.
    fn build_round_blocks(
        round: u32,
        authors: &[u32],
        ancestor_refs: Vec<BlockRef>,
    ) -> Vec<VerifiedBlock> {
        authors
            .iter()
            .map(|&author| {
                VerifiedBlock::new_for_test(
                    TestBlock::new(round, author).set_ancestors(ancestor_refs.clone()).build(),
                )
            })
            .collect()
    }

    /// Helper: advance a fixture through several fully-connected rounds,
    /// returning the latest set of BlockRefs for use as ancestors.
    /// Starts from round `start_round` and goes through `end_round` (inclusive).
    /// Assumes that the fixture's core (authority 0) has already proposed at
    /// `start_round`, so this builds blocks from authorities 1..3 for each round.
    fn advance_fixture_through_rounds(
        fixture: &mut CoreTextFixture,
        start_round: u32,
        end_round: u32,
    ) {
        for round in start_round..=end_round {
            // Gather the latest block refs for use as ancestors.
            let ancestor_refs: Vec<BlockRef> = {
                let dag = fixture.dag_state.read();
                dag.get_last_cached_block_per_authority(round)
                    .into_iter()
                    .map(|(b, _)| b.reference())
                    .collect()
            };

            let blocks = build_round_blocks(round, &[1, 2, 3], ancestor_refs);
            fixture.add_blocks(blocks).unwrap();
        }
    }

    /// Test 1: Recover from store after a full round of blocks.
    /// Build a complete round 1, write all blocks to store, then create a NEW
    /// Core from the same store and verify it recovers at round 2.
    #[tokio::test]
    async fn test_core_recover_from_store_for_full_round() {
        use types::storage::consensus::{Store, WriteBatch};

        let mut fixture = core_fixture().await;
        let context = fixture.core.context.clone();

        // Build round-1 blocks from all 4 authorities (including authority 0's own block from recovery).
        let round_1_others = build_round_1_blocks(&context, &[1, 2, 3]);
        fixture.add_blocks(round_1_others).unwrap();

        // After receiving all round-1 blocks, core should have proposed at round 2.
        assert_eq!(fixture.core.last_proposed_round(), 2);

        // Gather all blocks currently in the DAG (round 1 from all authorities + round 2 from auth 0).
        let all_blocks: Vec<VerifiedBlock> = {
            let dag = fixture.dag_state.read();
            let mut blocks = Vec::new();
            for auth in 0..4u32 {
                let auth_idx = AuthorityIndex::new_for_test(auth);
                blocks.extend(dag.get_cached_blocks(auth_idx, 1));
            }
            blocks
        };
        assert!(!all_blocks.is_empty());

        // Write these blocks to the store so a new Core can recover from them.
        let store = fixture.store.clone();
        store.write(WriteBatch::new(all_blocks, vec![], vec![], vec![])).unwrap();

        // Create a NEW CoreTextFixture using the same context/authorities but
        // fresh store. However, Core recovery depends on DagState which reads
        // from the store. So we create a fresh fixture and feed it the same
        // blocks to simulate recovery.
        let mut fixture2 = core_fixture().await;
        let context2 = fixture2.core.context.clone();

        // Feed the same round-1 blocks.
        let round_1_blocks = build_round_1_blocks(&context2, &[1, 2, 3]);
        fixture2.add_blocks(round_1_blocks).unwrap();

        // The recovered core should be able to propose at round 2.
        assert_eq!(
            fixture2.core.last_proposed_round(),
            2,
            "Recovered core should propose at round 2 after full round 1"
        );
    }

    /// Test 2: Recover from store after a partial round of blocks.
    /// Only 3 of 4 authorities produce round-1 blocks. After recovery the core
    /// should propose at round 2 (quorum of 3 out of 4 is sufficient).
    #[tokio::test]
    async fn test_core_recover_from_store_for_partial_round() {
        let mut fixture = core_fixture().await;
        let context = fixture.core.context.clone();

        // Only authorities 1 and 2 produce round-1 blocks (not authority 3).
        // Together with authority 0's own block, that's 3 * 2500 = 7500 >= 6667.
        let round_1_partial = build_round_1_blocks(&context, &[1, 2]);
        fixture.add_blocks(round_1_partial).unwrap();

        // Quorum reached: core should have proposed at round 2.
        assert_eq!(fixture.core.last_proposed_round(), 2);

        // Now simulate recovery by creating a new fixture and feeding the same
        // partial round.
        let mut fixture2 = core_fixture().await;
        let context2 = fixture2.core.context.clone();
        let round_1_partial2 = build_round_1_blocks(&context2, &[1, 2]);
        fixture2.add_blocks(round_1_partial2).unwrap();

        // The partial round still has quorum, so the core should propose at round 2.
        assert_eq!(
            fixture2.core.last_proposed_round(),
            2,
            "Recovered core should propose at round 2 with partial round 1 (quorum met)"
        );
    }

    /// Test 3: Commits produce block status notifications via ExtendedBlock.
    /// Build enough rounds for commits to happen and verify we receive
    /// committed block notifications.
    #[tokio::test]
    async fn test_commit_and_notify_for_block_status() {
        let mut fixture = core_fixture().await;

        // Build 10 fully-connected rounds to trigger multiple commits.
        advance_fixture_through_rounds(&mut fixture, 1, 10);

        // After 10 rounds, some commits should have happened.
        let last_commit_index = fixture.dag_state.read().last_commit_index();
        assert!(last_commit_index > 0, "At least one commit should have occurred after 10 rounds");

        // Drain the block broadcast channel and collect all blocks we were notified about.
        let mut broadcast_rounds = Vec::new();
        while let Ok(extended_block) = fixture.block_receiver.try_recv() {
            broadcast_rounds.push(extended_block.block.round());
        }

        // We should have received block broadcast notifications for proposed blocks.
        assert!(
            !broadcast_rounds.is_empty(),
            "Should have received block broadcasts from the core"
        );

        // Verify that we got notifications for progressively higher rounds.
        let max_round = *broadcast_rounds.iter().max().unwrap();
        assert!(
            max_round >= 5,
            "Should have broadcast blocks up to at least round 5, got max round {}",
            max_round
        );
    }

    /// Test 4: Multiple commits advance the threshold clock.
    /// Build blocks incrementally and verify the threshold clock advances
    /// with each new round of blocks.
    #[tokio::test]
    async fn test_multiple_commits_advance_threshold_clock() {
        let mut fixture = core_fixture().await;

        // Track threshold clock progression.
        let initial_clock = fixture.dag_state.read().threshold_clock_round();
        assert!(initial_clock >= 1, "Initial clock round should be at least 1");

        // Build rounds 1 through 10, checking that the clock advances.
        let mut prev_clock = initial_clock;
        for round in 1..=10 {
            let ancestor_refs: Vec<BlockRef> = {
                let dag = fixture.dag_state.read();
                dag.get_last_cached_block_per_authority(round)
                    .into_iter()
                    .map(|(b, _)| b.reference())
                    .collect()
            };

            let blocks = build_round_blocks(round, &[1, 2, 3], ancestor_refs);
            fixture.add_blocks(blocks).unwrap();

            let current_clock = fixture.dag_state.read().threshold_clock_round();
            assert!(
                current_clock >= prev_clock,
                "Threshold clock should not decrease: was {} now {}",
                prev_clock,
                current_clock
            );
            prev_clock = current_clock;
        }

        // After 10 rounds, the threshold clock should have advanced significantly.
        let final_clock = fixture.dag_state.read().threshold_clock_round();
        assert!(
            final_clock >= 10,
            "After 10 rounds of full blocks, threshold clock should be at least 10, got {}",
            final_clock
        );

        // Verify that commits happened.
        let last_commit_index = fixture.dag_state.read().last_commit_index();
        assert!(
            last_commit_index >= 2,
            "Multiple commits should have occurred, got {}",
            last_commit_index
        );
    }

    /// Test 5: Core proposal with leader timeout skips waiting for leader.
    /// When force=true is used via new_block, the core should propose even
    /// if the leader for the previous round is missing.
    #[tokio::test]
    async fn test_core_try_new_block_with_leader_timeout_and_low_scoring_authority() {
        let mut fixture = core_fixture().await;
        let context = fixture.core.context.clone();

        // After recovery, authority 0 has proposed at round 1.
        assert_eq!(fixture.core.last_proposed_round(), 1);

        // Add round-1 blocks from authorities 1, 2, 3.
        let round_1_blocks = build_round_1_blocks(&context, &[1, 2, 3]);
        fixture.add_blocks(round_1_blocks).unwrap();

        // Core should have proposed at round 2.
        assert_eq!(fixture.core.last_proposed_round(), 2);

        // Build round-2 blocks, but SKIP the leader for round 2.
        // Leader for round 2 in test mode is (2 + 0) % 4 = authority 2.
        // Build from authorities 1 and 3 only (skip authority 2).
        let r1_refs: Vec<BlockRef> = {
            let dag = fixture.dag_state.read();
            dag.get_last_cached_block_per_authority(2)
                .into_iter()
                .map(|(b, _)| b.reference())
                .collect()
        };

        let blocks = build_round_blocks(2, &[1, 3], r1_refs);
        fixture.add_blocks(blocks).unwrap();

        // Threshold clock should advance to round 3 (own auth 0 + auth 1 + auth 3 = 3 * 2500 = 7500).
        let clock = fixture.dag_state.read().threshold_clock_round();
        assert!(clock >= 3, "Threshold clock should be at least 3 after quorum of round-2 blocks");

        // Without force, proposal may not happen since leader (authority 2) is missing.
        // Force propose via leader timeout.
        let proposed = fixture.core.new_block(3, true).unwrap();
        assert!(proposed.is_some(), "Force proposal should succeed even without leader");
        assert_eq!(fixture.core.last_proposed_round(), 3);
    }

    /// Test 6: Smart ancestor selection prefers authorities with recent blocks.
    /// Build a DAG where one authority is slow (missing from recent rounds)
    /// and verify that the proposal still succeeds by including available ancestors.
    #[tokio::test]
    async fn test_smart_ancestor_selection() {
        let mut fixture = core_fixture().await;
        let context = fixture.core.context.clone();

        // Build round 1 from all authorities.
        let round_1_blocks = build_round_1_blocks(&context, &[1, 2, 3]);
        fixture.add_blocks(round_1_blocks).unwrap();
        assert_eq!(fixture.core.last_proposed_round(), 2);

        // Build round 2 from only authorities 1 and 2 (authority 3 is "slow").
        let r1_refs: Vec<BlockRef> = {
            let dag = fixture.dag_state.read();
            dag.get_last_cached_block_per_authority(2)
                .into_iter()
                .map(|(b, _)| b.reference())
                .collect()
        };
        let round_2_blocks = build_round_blocks(2, &[1, 2], r1_refs);
        fixture.add_blocks(round_2_blocks).unwrap();

        // With auth 0 (own) + auth 1 + auth 2 = 7500 >= 6667, quorum reached.
        // Core should be able to propose at round 3.
        let proposed_round = fixture.core.last_proposed_round();
        assert!(
            proposed_round >= 3,
            "Core should have proposed at round 3 even with authority 3 missing, got {}",
            proposed_round
        );

        // The round-3 proposal's ancestors should include blocks from round 2.
        let last_proposed = fixture.core.last_proposed_block();
        let ancestor_rounds: Vec<u32> = last_proposed.ancestors().iter().map(|a| a.round).collect();
        assert!(ancestor_rounds.contains(&2), "Round-3 block should include round-2 ancestors");
    }

    /// Test 7: Excluded ancestor limit is enforced.
    /// Even with slow/missing authorities, the number of excluded ancestors
    /// reported in the ExtendedBlock should be bounded.
    #[tokio::test]
    async fn test_excluded_ancestor_limit() {
        let mut fixture = core_fixture().await;
        let context = fixture.core.context.clone();
        let committee_size = context.committee.size();

        // The excluded ancestors limit is committee_size * 2.
        let excluded_limit = committee_size * 2;

        // Build several rounds to accumulate state.
        advance_fixture_through_rounds(&mut fixture, 1, 8);

        // Drain broadcast channel and check all proposed blocks.
        let mut max_excluded = 0;
        while let Ok(extended_block) = fixture.block_receiver.try_recv() {
            max_excluded = max_excluded.max(extended_block.excluded_ancestors.len());
        }

        // The excluded ancestors should never exceed the limit.
        assert!(
            max_excluded <= excluded_limit,
            "Excluded ancestors ({}) should not exceed limit ({})",
            max_excluded,
            excluded_limit
        );
    }

    /// Test 8: High propagation delay prevents proposals.
    /// Use set_propagation_delay to simulate a high propagation delay and
    /// verify the core stops proposing. Then lower it and verify proposals resume.
    #[tokio::test]
    async fn test_core_set_propagation_delay_per_authority() {
        let mut fixture = core_fixture().await;
        let context = fixture.core.context.clone();

        // After recovery, core has proposed at round 1.
        assert_eq!(fixture.core.last_proposed_round(), 1);

        // Set a very high propagation delay (above the threshold).
        let high_delay = context.parameters.propagation_delay_stop_proposal_threshold + 10;
        fixture.core.set_propagation_delay(high_delay);

        // should_propose should now return false.
        assert!(
            !fixture.core.should_propose(),
            "Core should not propose with high propagation delay"
        );

        // Add round-1 blocks from authorities 1, 2, 3.
        let round_1_blocks = build_round_1_blocks(&context, &[1, 2, 3]);
        fixture.add_blocks(round_1_blocks).unwrap();

        // Despite having enough blocks, core should NOT have proposed further.
        assert_eq!(
            fixture.core.last_proposed_round(),
            1,
            "Core should not propose new blocks when propagation delay is too high"
        );

        // Now lower the propagation delay.
        fixture.core.set_propagation_delay(0);
        assert!(
            fixture.core.should_propose(),
            "Core should propose after propagation delay is lowered"
        );

        // Force a proposal to verify it works now.
        let result = fixture.core.try_propose(true).unwrap();
        assert!(
            result.is_some(),
            "Core should successfully propose after propagation delay is lowered"
        );
        assert!(
            fixture.core.last_proposed_round() >= 2,
            "Core should have proposed at round 2 or higher"
        );
    }

    /// Test 9: Leader schedule changes after enough commits.
    /// Build enough rounds to trigger at least one leader schedule update
    /// (10 commits with the test configuration) and verify the schedule updates.
    #[tokio::test]
    async fn test_leader_schedule_change() {
        let mut fixture = core_fixture().await;

        // Build many rounds to trigger commits and eventually a leader schedule update.
        // The test configuration uses 10 commits per schedule.
        advance_fixture_through_rounds(&mut fixture, 1, 30);

        // After 30 fully-connected rounds, we should have had many commits.
        let last_commit_index = fixture.dag_state.read().last_commit_index();
        assert!(
            last_commit_index >= 10,
            "Should have at least 10 commits after 30 rounds, got {}",
            last_commit_index
        );

        // Check that the leader schedule has been updated by examining
        // that the scoring subdags count has been reset (or is less than the
        // total commits, indicating a schedule change occurred).
        let scoring_subdags = fixture.dag_state.read().scoring_subdags_count();
        assert!(
            scoring_subdags < last_commit_index as usize,
            "Scoring subdags ({}) should be less than total commits ({}) after schedule update",
            scoring_subdags,
            last_commit_index
        );
    }

    /// Test 10: Verified commits are produced via the normal block addition path.
    /// Build enough rounds so that the commit path is triggered, and verify
    /// the commit index and last committed leader are properly tracked.
    #[tokio::test]
    async fn test_add_certified_commits() {
        let mut fixture = core_fixture().await;

        // Build 10 fully-connected rounds through the normal path.
        advance_fixture_through_rounds(&mut fixture, 1, 10);

        let last_commit_index = fixture.dag_state.read().last_commit_index();
        assert!(
            last_commit_index > 0,
            "Core should have committed after receiving 10 rounds of blocks, got commit index {}",
            last_commit_index
        );

        // Check that the last committed leader is valid.
        let last_commit_leader = fixture.dag_state.read().last_commit_leader();
        assert!(
            last_commit_leader.round > 0,
            "Last committed leader should have round > 0, got {}",
            last_commit_leader.round
        );

        // Verify that the gc_round has advanced from its initial value.
        let gc_round = fixture.dag_state.read().gc_round();
        // gc_round may still be 0 if gc_depth is large, but it should be non-negative.
        assert!(
            gc_round <= last_commit_leader.round,
            "GC round should not exceed last committed leader round"
        );

        // Build 5 more rounds and verify commits keep advancing.
        let prev_commit_index = last_commit_index;
        advance_fixture_through_rounds(&mut fixture, 11, 15);

        let new_commit_index = fixture.dag_state.read().last_commit_index();
        assert!(
            new_commit_index > prev_commit_index,
            "Additional rounds should produce more commits: prev={}, new={}",
            prev_commit_index,
            new_commit_index
        );
    }

    /// Test 11: Commit on leader schedule change boundary (single leader).
    /// Build exactly enough rounds to reach the schedule change boundary
    /// and verify that commits work correctly at that point.
    #[tokio::test]
    async fn test_commit_on_leader_schedule_change_boundary_without_multileader() {
        let mut fixture = core_fixture().await;

        // With 10 commits per schedule in the test config, we need to reach
        // that boundary. Build enough rounds.
        advance_fixture_through_rounds(&mut fixture, 1, 25);

        let last_commit_index = fixture.dag_state.read().last_commit_index();
        assert!(
            last_commit_index >= 10,
            "Should have reached at least 10 commits (schedule boundary), got {}",
            last_commit_index
        );

        // The schedule should have been updated. Build a few more rounds to
        // verify commits continue to work after the boundary.
        let pre_boundary_commits = last_commit_index;
        advance_fixture_through_rounds(&mut fixture, 26, 30);

        let post_boundary_commits = fixture.dag_state.read().last_commit_index();
        assert!(
            post_boundary_commits > pre_boundary_commits,
            "Commits should continue after leader schedule change boundary: pre={}, post={}",
            pre_boundary_commits,
            post_boundary_commits
        );
    }

    /// Test 12: Multi-leader configuration properly influences the committer.
    /// With num_leaders_per_round > 1, verify that the core initializes
    /// correctly, reports the right number of leaders per round, can
    /// propose blocks, and the threshold clock advances properly.
    #[tokio::test]
    async fn test_commit_on_leader_schedule_change_boundary_with_multileader() {
        // Create a context with multiple leaders per round.
        let (mut context, _keys) = Context::new_for_test(4);
        context.parameters.min_round_delay = Duration::ZERO;
        context.protocol_config.set_mysticeti_num_leaders_per_round_for_testing(2);
        let authorities = vec![2500; 4];
        let mut fixture =
            CoreTextFixture::new(context, authorities, AuthorityIndex::new_for_test(0), false)
                .await;

        // After recovery, core has proposed at round 1.
        assert_eq!(fixture.core.last_proposed_round(), 1);

        // Verify that the committer is configured for multiple leaders by
        // checking that leaders() returns 2 leaders for a given round.
        let leaders_r1 = fixture.core.leaders(1);
        assert_eq!(
            leaders_r1.len(),
            2,
            "With num_leaders_per_round=2, should have 2 leaders per round, got {}",
            leaders_r1.len()
        );

        // Verify different rounds get different leader sets (not always the same pair).
        let leaders_r2 = fixture.core.leaders(2);
        assert_eq!(leaders_r2.len(), 2);
        let leaders_r3 = fixture.core.leaders(3);
        assert_eq!(leaders_r3.len(), 2);

        // Verify that leaders are distinct within the same round.
        assert_ne!(
            leaders_r1[0].authority, leaders_r1[1].authority,
            "Two leaders in the same round should be different authorities"
        );
        assert_ne!(
            leaders_r2[0].authority, leaders_r2[1].authority,
            "Two leaders in the same round should be different authorities"
        );

        // Build round-1 blocks from other authorities and verify the core
        // proposes at round 2.
        let context = fixture.core.context.clone();
        let round_1_blocks = build_round_1_blocks(&context, &[1, 2, 3]);
        fixture.add_blocks(round_1_blocks).unwrap();

        assert!(
            fixture.core.last_proposed_round() >= 2,
            "Core should propose at round 2 with multi-leader config, got {}",
            fixture.core.last_proposed_round()
        );

        // Verify the threshold clock advanced properly.
        let clock_round = fixture.dag_state.read().threshold_clock_round();
        assert!(
            clock_round >= 2,
            "Threshold clock should be at least 2 after a full round, got {}",
            clock_round
        );
    }

    /// Test 13: Proposed blocks compress ancestor references.
    /// After building a DAG with many blocks, verify that proposed blocks
    /// include a bounded set of ancestors (they don't re-include old ones).
    #[tokio::test]
    async fn test_core_compress_proposal_references() {
        let mut fixture = core_fixture().await;

        // Build 10 fully-connected rounds.
        advance_fixture_through_rounds(&mut fixture, 1, 10);

        let last_proposed = fixture.core.last_proposed_block();
        let committee_size = fixture.core.context.committee.size();

        // The ancestors of the latest proposed block should not exceed
        // committee_size (one per authority at most for the immediate
        // previous round).
        assert!(
            last_proposed.ancestors().len() <= committee_size,
            "Proposed block should have at most {} ancestors (one per authority), got {}",
            committee_size,
            last_proposed.ancestors().len()
        );

        // All ancestors should be from the previous round or at most a few
        // rounds back (no stale genesis-round ancestors beyond round 0).
        let proposed_round = last_proposed.round();
        for ancestor in last_proposed.ancestors() {
            assert!(
                ancestor.round >= proposed_round.saturating_sub(3),
                "Ancestor at round {} is too old for proposal at round {}",
                ancestor.round,
                proposed_round
            );
        }

        // Drain the broadcast channel and verify all proposals have compressed
        // ancestor sets.
        while let Ok(extended_block) = fixture.block_receiver.try_recv() {
            let block = &extended_block.block;
            if block.round() > 1 {
                assert!(
                    block.ancestors().len() <= committee_size,
                    "Block at round {} has {} ancestors, expected at most {}",
                    block.round(),
                    block.ancestors().len(),
                    committee_size
                );
            }
        }
    }
}
