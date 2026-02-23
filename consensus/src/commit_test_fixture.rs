// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

// Test fixture infrastructure for randomized consensus commit tests.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ops::Bound::Included;
use std::sync::Arc;

use parking_lot::RwLock;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use types::committee::{AuthorityIndex, VotingPower};
use types::consensus::{
    block::{
        BlockAPI, BlockDigest, BlockRef, BlockTransactionVotes, Round, Slot, TestBlock,
        Transaction, TransactionIndex, VerifiedBlock, genesis_blocks,
    },
    commit::CommittedSubDag,
    context::Context,
};
use types::storage::consensus::mem_store::MemStore;

use crate::{
    block_manager::BlockManager,
    dag_state::DagState,
    leader_schedule::{LeaderSchedule, LeaderSwapTable},
    linearizer::Linearizer,
    test_dag_builder::DagBuilder,
    universal_committer::{
        UniversalCommitter, universal_committer_builder::UniversalCommitterBuilder,
    },
};

/// Test fixture that bundles all the consensus components needed for commit testing:
/// DagState, UniversalCommitter, Linearizer, BlockManager.
pub(crate) struct CommitTestFixture {
    pub dag_state: Arc<RwLock<DagState>>,
    pub committer: UniversalCommitter,
    pub linearizer: Linearizer,
    pub block_manager: BlockManager,
}

impl CommitTestFixture {
    /// Create a new test fixture from a given context.
    pub fn new(context: Arc<Context>) -> Self {
        let dag_state =
            Arc::new(RwLock::new(DagState::new(context.clone(), Arc::new(MemStore::new()))));
        let leader_schedule =
            Arc::new(LeaderSchedule::new(context.clone(), LeaderSwapTable::default()));
        let committer =
            UniversalCommitterBuilder::new(context.clone(), leader_schedule, dag_state.clone())
                .with_pipeline(true)
                .build();
        let linearizer = Linearizer::new(context.clone(), dag_state.clone());
        let block_manager = BlockManager::new(context.clone(), dag_state.clone());

        Self { dag_state, committer, linearizer, block_manager }
    }

    /// Create a Context configured for testing with the given parameters.
    pub fn context_with_options(
        num_authorities: usize,
        authority_index: u32,
        _gc_depth: u32,
    ) -> Arc<Context> {
        let (context, _keypairs) = Context::new_for_test(num_authorities);
        let context = context.with_authority_index(AuthorityIndex::new_for_test(authority_index));
        Arc::new(context)
    }

    /// Try to accept blocks through the block_manager, which validates causal history,
    /// then adds accepted blocks to dag_state.
    pub fn try_accept_blocks(&mut self, blocks: Vec<VerifiedBlock>) {
        let (accepted_blocks, _missing) = self.block_manager.try_accept_blocks(blocks);
        if !accepted_blocks.is_empty() {
            self.dag_state.write().accept_blocks(accepted_blocks);
        }
    }

    /// Add blocks directly to dag_state, bypassing the block_manager.
    pub fn add_blocks(&mut self, blocks: Vec<VerifiedBlock>) {
        self.dag_state.write().accept_blocks(blocks);
    }

    /// Run try_decide on the committer, then process committed leaders through
    /// the linearizer to produce committed sub-dags.
    pub fn try_commit(&mut self, last_decided: Slot) -> Vec<CommittedSubDag> {
        let decided_leaders = self.committer.try_decide(last_decided);
        let committed_blocks: Vec<VerifiedBlock> = decided_leaders
            .into_iter()
            .filter_map(|leader| leader.into_committed_block())
            .collect();
        self.linearizer.handle_commit(committed_blocks)
    }

    /// Run try_decide on the committer, returning both the committed sub-dags and
    /// the updated last_decided slot. This is useful for incremental commit tracking
    /// where blocks are delivered one at a time.
    pub fn try_commit_tracking(&mut self, last_decided: Slot) -> (Vec<CommittedSubDag>, Slot) {
        let decided_leaders = self.committer.try_decide(last_decided);
        let new_last_decided =
            decided_leaders.last().map(|leader| leader.slot()).unwrap_or(last_decided);
        let committed_blocks: Vec<VerifiedBlock> = decided_leaders
            .into_iter()
            .filter_map(|leader| leader.into_committed_block())
            .collect();
        let committed = self.linearizer.handle_commit(committed_blocks);
        (committed, new_last_decided)
    }

    /// Check that the block_manager has no suspended blocks.
    pub fn has_no_suspended_blocks(&self) -> bool {
        self.block_manager.is_empty()
    }
}

/// A wrapper around DagBuilder that allows blocks to be delivered in random order.
pub(crate) struct RandomDag {
    pub blocks: Vec<VerifiedBlock>,
}

impl RandomDag {
    /// Create a new RandomDag from a DagBuilder's blocks.
    pub fn new(dag_builder: &DagBuilder) -> Self {
        Self { blocks: dag_builder.all_blocks() }
    }

    /// Create an iterator that delivers blocks in a random order determined by the seed.
    pub fn random_iter(&self, seed: u64) -> RandomDagIterator {
        let mut blocks = self.blocks.clone();
        let mut rng = StdRng::seed_from_u64(seed);
        blocks.shuffle(&mut rng);
        RandomDagIterator { blocks }
    }

    /// Return all blocks in the natural (round-ascending) order.
    pub fn blocks_in_order(&self) -> Vec<VerifiedBlock> {
        let mut blocks = self.blocks.clone();
        blocks.sort_by_key(|b| (b.round(), b.author().value()));
        blocks
    }
}

/// Iterator that yields blocks from a RandomDag in a shuffled order.
pub(crate) struct RandomDagIterator {
    blocks: Vec<VerifiedBlock>,
}

impl Iterator for RandomDagIterator {
    type Item = VerifiedBlock;

    fn next(&mut self) -> Option<Self::Item> {
        if self.blocks.is_empty() { None } else { Some(self.blocks.remove(0)) }
    }
}

/// Assert that multiple commit sequences contain the same committed leaders
/// in the same order. Compares by slot (round + authority).
pub(crate) fn assert_commit_sequences_match(sequences: &[Vec<CommittedSubDag>]) {
    assert!(sequences.len() >= 2, "Need at least 2 sequences to compare");

    let reference = &sequences[0];
    for (i, sequence) in sequences.iter().enumerate().skip(1) {
        assert_eq!(
            reference.len(),
            sequence.len(),
            "Commit sequence lengths differ: reference has {}, sequence {} has {}",
            reference.len(),
            i,
            sequence.len()
        );
        for (j, (ref_subdag, cmp_subdag)) in reference.iter().zip(sequence.iter()).enumerate() {
            assert_eq!(
                ref_subdag.leader, cmp_subdag.leader,
                "Commit sequence mismatch at position {}: reference leader {:?} != sequence {} leader {:?}",
                j, ref_subdag.leader, i, cmp_subdag.leader
            );
        }
    }
}

// ---- RandomDagConfig, EquivocatingRandomDag, ConstrainedRandomDagIterator ----

/// Configuration for generating a randomized DAG with optional equivocators
/// and reject votes.
pub(crate) struct RandomDagConfig {
    /// Number of distinct authorities creating blocks.
    pub num_authorities: usize,
    /// Number of rounds to generate.
    pub num_rounds: Round,
    /// Number of transactions per block.
    pub num_transactions: u32,
    /// Percentage chance (0-100) of each transaction being rejected.
    pub reject_percentage: u8,
    /// Each element specifies the authority index and the number of extra
    /// equivocating instances for that authority.
    pub equivocators: Vec<(AuthorityIndex, u16)>,
}

/// Identifies a consensus instance by authority index and an instance number
/// to differentiate between equivocators. Instance 0 is the honest instance.
type InstanceID = (AuthorityIndex, u16);

/// A randomly generated DAG for testing commit patterns with equivocators
/// and reject votes. Blocks are built round by round with quorum-based
/// ancestor selection and optional equivocating blocks.
pub(crate) struct EquivocatingRandomDag {
    context: Arc<Context>,
    pub blocks: Vec<VerifiedBlock>,
    num_rounds: Round,
}

impl EquivocatingRandomDag {
    /// Creates a new EquivocatingRandomDag with generated blocks.
    pub fn new(context: Arc<Context>, rng: &mut StdRng, config: RandomDagConfig) -> Self {
        let RandomDagConfig {
            num_authorities: _,
            num_rounds,
            num_transactions,
            reject_percentage,
            equivocators,
        } = config;

        let committee = &context.committee;
        let quorum_threshold = committee.quorum_threshold();
        let total_stake: VotingPower = committee.total_votes();

        // Create instance ID for each authority and equivocators.
        let mut instances: Vec<InstanceID> =
            (0..committee.size()).map(|i| (AuthorityIndex::new_for_test(i as u32), 0u16)).collect();
        for (authority, num_equivocators) in equivocators {
            for i in 1..=num_equivocators {
                instances.push((authority, i));
            }
        }

        let genesis = genesis_blocks(&context);
        let genesis_by_author: BTreeMap<AuthorityIndex, VerifiedBlock> =
            genesis.iter().map(|b| (b.author(), b.clone())).collect();

        // Store all blocks for lookup and range search.
        let mut all_blocks: BTreeMap<BlockRef, VerifiedBlock> = BTreeMap::new();
        for block in &genesis {
            all_blocks.insert(block.reference(), block.clone());
        }

        // Track the latest block per instance. Equivocators start from the
        // same genesis block per authority.
        let mut latest_blocks: BTreeMap<InstanceID, VerifiedBlock> = instances
            .iter()
            .map(|&(a, i)| {
                let b = genesis_by_author.get(&a).unwrap();
                ((a, i), b.clone())
            })
            .collect();

        // Track included block refs per instance (for reject vote generation).
        let mut included_refs: BTreeMap<InstanceID, BTreeSet<BlockRef>> = BTreeMap::new();

        for r in 1..=num_rounds {
            // Select random quorum-or-more stake to produce blocks this round.
            let target_stake = rng.gen_range(quorum_threshold..=total_stake);

            // Select random instances to produce blocks this round.
            let mut proposers = instances.clone();
            proposers.shuffle(rng);
            let mut selected_stake: VotingPower = 0;
            let mut selected_authorities = vec![false; committee.size()];
            let selected_proposers: Vec<_> = proposers
                .into_iter()
                .take_while(|instance| {
                    if selected_stake >= target_stake {
                        return false;
                    }
                    if !selected_authorities[instance.0.value()] {
                        selected_authorities[instance.0.value()] = true;
                        selected_stake += committee.stake_by_index(instance.0);
                    }
                    true
                })
                .collect();

            let mut current_round_blocks = Vec::new();
            for instance_id in selected_proposers {
                let block = build_block_for_instance(
                    &context,
                    &instances,
                    rng,
                    r,
                    instance_id,
                    num_transactions,
                    reject_percentage,
                    &all_blocks,
                    &mut latest_blocks,
                    &mut included_refs,
                );
                current_round_blocks.push((instance_id, block));
            }

            // Update state with current round blocks.
            for (instance_id, block) in current_round_blocks {
                all_blocks.insert(block.reference(), block.clone());
                latest_blocks.insert(instance_id, block);
            }
        }

        EquivocatingRandomDag {
            context,
            blocks: all_blocks.values().cloned().collect(),
            num_rounds,
        }
    }

    /// Creates an iterator yielding blocks in constrained random order.
    /// `max_step` limits how far ahead of the current quorum round a block
    /// can be delivered.
    pub fn random_iter<'a>(
        &'a self,
        rng: &'a mut StdRng,
        max_step: Round,
    ) -> ConstrainedRandomDagIterator<'a> {
        ConstrainedRandomDagIterator::new(self, rng, max_step)
    }
}

/// Builds a single block for the given consensus instance at the specified round.
#[allow(clippy::too_many_arguments)]
fn build_block_for_instance(
    context: &Arc<Context>,
    instances: &[InstanceID],
    rng: &mut StdRng,
    round: Round,
    own_instance: InstanceID,
    num_transactions: u32,
    reject_percentage: u8,
    all_blocks: &BTreeMap<BlockRef, VerifiedBlock>,
    latest_blocks: &mut BTreeMap<InstanceID, VerifiedBlock>,
    included_refs: &mut BTreeMap<InstanceID, BTreeSet<BlockRef>>,
) -> VerifiedBlock {
    let committee = &context.committee;
    let quorum_threshold = committee.quorum_threshold();
    let own_authority = own_instance.0;

    // Select blocks from the previous round until quorum stake is reached.
    let prev_round = round - 1;
    let mut prev_round_blocks: Vec<_> = all_blocks
        .range((
            Included(BlockRef::new(prev_round, AuthorityIndex::new_for_test(0), BlockDigest::MIN)),
            Included(BlockRef::new(
                prev_round,
                AuthorityIndex::new_for_test(u32::MAX),
                BlockDigest::MAX,
            )),
        ))
        .map(|(_, b)| b.clone())
        .collect();
    prev_round_blocks.shuffle(rng);

    let mut parent_stake: VotingPower = 0;
    let mut selected_authorities = vec![false; committee.size()];
    let quorum_parents: Vec<_> = prev_round_blocks
        .into_iter()
        .filter(|b| {
            if parent_stake >= quorum_threshold {
                return false;
            }
            if selected_authorities[b.author().value()] {
                return false;
            }
            selected_authorities[b.author().value()] = true;
            parent_stake += committee.stake_by_index(b.author());
            true
        })
        .collect();

    // Find unselected instances and optionally add extra ancestors.
    let mut unselected_instances: Vec<_> = instances
        .iter()
        .filter(|(authority, _)| !selected_authorities[authority.value()])
        .cloned()
        .collect();
    unselected_instances.shuffle(rng);

    // Use min of two uniform samples to bias toward fewer additional ancestors.
    let extra_count = rng
        .gen_range(0..=unselected_instances.len())
        .min(rng.gen_range(0..=unselected_instances.len()));
    let additional_ancestor_blocks: Vec<_> = unselected_instances[0..extra_count]
        .iter()
        .filter_map(|&(authority, instance)| {
            if selected_authorities[authority.value()] {
                return None;
            }
            let block = latest_blocks.get(&(authority, instance))?;
            assert!(
                block.round() < round,
                "latest_blocks should only contain blocks from previous rounds"
            );
            selected_authorities[authority.value()] = true;
            Some(block.clone())
        })
        .collect();

    // Combine ancestors.
    let mut ancestor_blocks = quorum_parents;
    ancestor_blocks.extend(additional_ancestor_blocks);
    if !ancestor_blocks.iter().any(|b| b.author() == own_authority) {
        ancestor_blocks.push(latest_blocks[&own_instance].clone());
    }
    let ancestors: Vec<_> = ancestor_blocks.iter().map(|b| b.reference()).collect();

    // Find newly connected blocks via BFS (for reject vote generation).
    let mut newly_connected = Vec::new();
    let mut queue = VecDeque::from_iter(ancestors.iter().copied());
    while let Some(block_ref) = queue.pop_front() {
        if block_ref.round == 0 {
            continue;
        }
        if included_refs.entry(own_instance).or_default().contains(&block_ref) {
            continue;
        }
        included_refs.entry(own_instance).or_default().insert(block_ref);
        newly_connected.push(block_ref);
        if let Some(block) = all_blocks.get(&block_ref) {
            queue.extend(block.ancestors().iter().copied());
        }
    }

    // Generate random reject votes for newly connected blocks.
    let votes: Vec<_> = newly_connected
        .iter()
        .filter(|_| reject_percentage > 0)
        .filter_map(|&block_ref| {
            let rejects: Vec<_> = (0..num_transactions)
                .filter(|_| rng.gen_range(0u8..100) < reject_percentage)
                .map(|idx| idx as TransactionIndex)
                .collect();
            if rejects.is_empty() {
                None
            } else {
                Some(BlockTransactionVotes { block_ref, rejects })
            }
        })
        .collect();

    let transactions: Vec<_> =
        (0..num_transactions).map(|_| Transaction::new(vec![1_u8; 16])).collect();

    let timestamp =
        (round as u64) * 1000 + (own_authority.value() as u64) + rng.gen_range(0u64..100);

    VerifiedBlock::new_for_test(
        TestBlock::new(round, own_authority.value() as u32)
            .set_transactions(transactions)
            .set_transaction_votes(votes)
            .set_ancestors(ancestors)
            .set_timestamp_ms(timestamp)
            .build(),
    )
}

/// Per-round state for the constrained random iteration.
#[derive(Clone, Default)]
struct RoundState {
    /// Total stake of visited blocks in this round.
    visited_stake: VotingPower,
    /// Indices of unvisited blocks in this round.
    unvisited: Vec<usize>,
}

/// Iterator that yields blocks in constrained random order. Selects from rounds
/// `completed_round + 1` to `quorum_round + max_step`, simulating arrival with
/// delays. This is ported from Sui's `RandomDagIterator`.
pub(crate) struct ConstrainedRandomDagIterator<'a> {
    dag: &'a EquivocatingRandomDag,
    rng: &'a mut StdRng,
    quorum_threshold: VotingPower,
    max_step: Round,
    /// Highest round where all prior rounds have quorum stake visited.
    quorum_round: Round,
    /// Highest round where all prior rounds have all blocks visited.
    completed_round: Round,
    /// State of each round.
    round_states: Vec<RoundState>,
    /// Number of blocks remaining to visit.
    num_remaining: usize,
}

impl<'a> ConstrainedRandomDagIterator<'a> {
    fn new(dag: &'a EquivocatingRandomDag, rng: &'a mut StdRng, max_step: Round) -> Self {
        let num_rounds = dag.num_rounds as usize;
        let committee = &dag.context.committee;
        let quorum_threshold = committee.quorum_threshold();

        let mut round_states: Vec<RoundState> = vec![RoundState::default(); num_rounds + 1];

        for (idx, block) in dag.blocks.iter().enumerate() {
            let round = block.round() as usize;
            if round > 0 && round <= num_rounds {
                round_states[round].unvisited.push(idx);
            }
        }

        let num_remaining: usize = round_states.iter().map(|s| s.unvisited.len()).sum();

        Self {
            dag,
            rng,
            max_step,
            quorum_round: 0,
            completed_round: 0,
            quorum_threshold,
            round_states,
            num_remaining,
        }
    }
}

impl Iterator for ConstrainedRandomDagIterator<'_> {
    type Item = VerifiedBlock;

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_remaining == 0 {
            return None;
        }

        // Eligible rounds: from first unvisited to quorum_round + max_step.
        let min_round = self.completed_round as usize + 1;
        let max_round =
            ((self.quorum_round + self.max_step) as usize).min(self.round_states.len() - 1);

        if min_round > max_round {
            return None;
        }

        let eligible_rounds = min_round..=max_round;

        let total_candidates: usize =
            eligible_rounds.clone().map(|r| self.round_states[r].unvisited.len()).sum();

        if total_candidates == 0 {
            return None;
        }

        // Select random candidate by index across eligible rounds.
        let mut selection = self.rng.gen_range(0..total_candidates);
        let mut selected_round = 0;
        let mut selected_pos = 0;

        for r in eligible_rounds {
            let count = self.round_states[r].unvisited.len();
            if selection < count {
                selected_round = r;
                selected_pos = selection;
                break;
            }
            selection -= count;
        }

        // Get block index and remove from unvisited.
        let block_idx = self.round_states[selected_round].unvisited.swap_remove(selected_pos);
        let block = self.dag.blocks[block_idx].clone();

        // Update visited stake for this round.
        let stake = self.dag.context.committee.stake_by_index(block.author());
        self.round_states[selected_round].visited_stake += stake;
        self.num_remaining -= 1;

        // Advance completed_round while next round has all blocks visited.
        while self
            .round_states
            .get(self.completed_round as usize + 1)
            .is_some_and(|s| s.unvisited.is_empty())
        {
            self.completed_round += 1;
        }

        // Advance quorum_round while next round has quorum stake visited.
        while self
            .round_states
            .get(self.quorum_round as usize + 1)
            .is_some_and(|s| s.visited_stake >= self.quorum_threshold)
        {
            self.quorum_round += 1;
        }

        Some(block)
    }
}
