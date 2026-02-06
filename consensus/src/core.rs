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
                                if  last_block_ref.round >= ancestor.round() {
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
