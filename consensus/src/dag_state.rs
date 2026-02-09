use std::{
    cmp::max,
    collections::{BTreeMap, BTreeSet, VecDeque},
    ops::Bound::{Excluded, Included, Unbounded},
    panic,
    sync::Arc,
    time::Duration,
    vec,
};

use itertools::Itertools as _;
use tokio::time::Instant;
use tracing::{debug, error, info, trace};
use types::committee::AuthorityIndex;
use types::consensus::{
    block::{
        BlockAPI, BlockDigest, BlockRef, BlockTimestampMs, GENESIS_ROUND, Round, Slot,
        TransactionIndex, VerifiedBlock, genesis_blocks,
    },
    commit::{
        CommitAPI as _, CommitDigest, CommitIndex, CommitInfo, CommitRef, CommitVote,
        CommittedSubDag, GENESIS_COMMIT_INDEX, TrustedCommit, load_committed_subdag_from_store,
    },
    context::Context,
    leader_scoring::{ReputationScores, ScoringSubdag},
};
use types::storage::consensus::{Store, WriteBatch};

use crate::threshold_clock::ThresholdClock;

/// DagState provides the API to write and read accepted blocks from the DAG.
/// Only uncommitted and last committed blocks are cached in memory.
/// The rest of blocks are stored on disk.
/// Refs to cached blocks and additional refs are cached as well, to speed up existence checks.
///
/// Note: DagState should be wrapped with Arc<parking_lot::RwLock<_>>, to allow
/// concurrent access from multiple components.
pub struct DagState {
    context: Arc<Context>,

    // The genesis blocks
    genesis: BTreeMap<BlockRef, VerifiedBlock>,

    // Contains recent blocks within CACHED_ROUNDS from the last committed round per authority.
    // Note: all uncommitted blocks are kept in memory.
    //
    // When GC is enabled, this map has a different semantic. It holds all the recent data for each authority making sure that it always have available
    // CACHED_ROUNDS worth of data. The entries are evicted based on the latest GC round, however the eviction process will respect the CACHED_ROUNDS.
    // For each authority, blocks are only evicted when their round is less than or equal to both `gc_round`, and `highest authority round - cached rounds`.
    // This ensures that the GC requirements are respected (we never clean up any block above `gc_round`), and there are enough blocks cached.
    recent_blocks: BTreeMap<BlockRef, BlockInfo>,

    // Indexes recent block refs by their authorities.
    // Vec position corresponds to the authority index.
    recent_refs_by_authority: Vec<BTreeSet<BlockRef>>,

    // Keeps track of the threshold clock for proposing blocks.
    threshold_clock: ThresholdClock,

    // Keeps track of the highest round that has been evicted for each authority. Any blocks that are of round <= evict_round
    // should be considered evicted, and if any exist we should not consider the causauly complete in the order they appear.
    // The `evicted_rounds` size should be the same as the committee size.
    evicted_rounds: Vec<Round>,

    // Highest round of blocks accepted.
    highest_accepted_round: Round,

    // Last consensus commit of the dag.
    last_commit: Option<TrustedCommit>,

    // Last wall time when commit round advanced. Does not persist across restarts.
    last_commit_round_advancement_time: Option<std::time::Instant>,

    // Last committed rounds per authority.
    last_committed_rounds: Vec<Round>,

    /// The committed subdags that have been scored but scores have not been used
    /// for leader schedule yet.
    scoring_subdag: ScoringSubdag,

    // Commit votes pending to be included in new blocks.
    // TODO: limit to 1st commit per round with multi-leader.
    // TODO: recover unproposed pending commit votes at startup.
    pending_commit_votes: VecDeque<CommitVote>,

    // Blocks and commits must be buffered for persistence before they can be
    // inserted into the local DAG or sent to output.
    blocks_to_write: Vec<VerifiedBlock>,
    commits_to_write: Vec<TrustedCommit>,

    // Buffers the reputation scores & last_committed_rounds to be flushed with the
    // next dag state flush. Not writing eagerly is okay because we can recover reputation scores
    // & last_committed_rounds from the commits as needed.
    commit_info_to_write: Vec<(CommitRef, CommitInfo)>,

    // Buffers finalized commits and their rejected transactions to be written to storage.
    finalized_commits_to_write: Vec<(CommitRef, BTreeMap<BlockRef, Vec<TransactionIndex>>)>,

    // Persistent storage for blocks, commits and other consensus data.
    store: Arc<dyn Store>,

    // The number of cached rounds
    cached_rounds: Round,
}

impl DagState {
    /// Initializes DagState from storage.
    pub fn new(context: Arc<Context>, store: Arc<dyn Store>) -> Self {
        let cached_rounds = context.parameters.dag_state_cached_rounds as Round;
        let num_authorities = context.committee.size();

        let genesis = genesis_blocks(context.as_ref())
            .into_iter()
            .map(|block| (block.reference(), block))
            .collect();

        let threshold_clock = ThresholdClock::new(1, context.clone());

        let last_commit = store
            .read_last_commit()
            .unwrap_or_else(|e| panic!("Failed to read from storage: {:?}", e));

        let commit_info = store
            .read_last_commit_info()
            .unwrap_or_else(|e| panic!("Failed to read from storage: {:?}", e));
        let (mut last_committed_rounds, commit_recovery_start_index) =
            if let Some((commit_ref, commit_info)) = commit_info {
                tracing::info!("Recovering committed state from {commit_ref} {commit_info:?}");
                (commit_info.committed_rounds, commit_ref.index + 1)
            } else {
                tracing::info!("Found no stored CommitInfo to recover from");
                (vec![0; num_authorities], GENESIS_COMMIT_INDEX + 1)
            };

        let mut unscored_committed_subdags = Vec::new();
        let mut scoring_subdag = ScoringSubdag::new(context.clone());

        if let Some(last_commit) = last_commit.as_ref() {
            store
                .scan_commits((commit_recovery_start_index..=last_commit.index()).into())
                .unwrap_or_else(|e| panic!("Failed to read from storage: {:?}", e))
                .iter()
                .for_each(|commit| {
                    for block_ref in commit.blocks() {
                        last_committed_rounds[block_ref.author] =
                            max(last_committed_rounds[block_ref.author], block_ref.round);
                    }
                    let committed_subdag =
                        load_committed_subdag_from_store(store.as_ref(), commit.clone(), vec![]);
                    // We don't need to recover reputation scores for unscored_committed_subdags
                    unscored_committed_subdags.push(committed_subdag);
                });
        }

        tracing::info!(
            "DagState was initialized with the following state: \
            {last_commit:?}; {last_committed_rounds:?}; {} unscored committed subdags;",
            unscored_committed_subdags.len()
        );

        scoring_subdag.add_subdags(std::mem::take(&mut unscored_committed_subdags));

        let mut state = Self {
            context: context.clone(),
            genesis,
            recent_blocks: BTreeMap::new(),
            recent_refs_by_authority: vec![BTreeSet::new(); num_authorities],
            threshold_clock,
            highest_accepted_round: 0,
            last_commit: last_commit.clone(),
            last_commit_round_advancement_time: None,
            last_committed_rounds: last_committed_rounds.clone(),
            pending_commit_votes: VecDeque::new(),
            blocks_to_write: vec![],
            commits_to_write: vec![],
            commit_info_to_write: vec![],
            finalized_commits_to_write: vec![],
            scoring_subdag,
            store: store.clone(),
            cached_rounds,
            evicted_rounds: vec![0; num_authorities],
        };

        for (authority_index, _) in context.committee.authorities() {
            let (blocks, eviction_round) = {
                // Find the latest block for the authority to calculate the eviction round. Then we want to scan and load the blocks from the eviction round and onwards only.
                // As reminder, the eviction round is taking into account the gc_round.
                let last_block = state
                    .store
                    .scan_last_blocks_by_author(authority_index, 1, None)
                    .expect("Database error");
                let last_block_round =
                    last_block.last().map(|b| b.round()).unwrap_or(GENESIS_ROUND);

                let eviction_round =
                    Self::eviction_round(last_block_round, state.gc_round(), state.cached_rounds);
                let blocks = state
                    .store
                    .scan_blocks_by_author(authority_index, eviction_round + 1)
                    .expect("Database error");

                (blocks, eviction_round)
            };

            state.evicted_rounds[authority_index] = eviction_round;

            // Update the block metadata for the authority.
            for block in &blocks {
                state.update_block_metadata(block);
            }

            debug!(
                "Recovered blocks {}: {:?}",
                authority_index,
                blocks.iter().map(|b| b.reference()).collect::<Vec<BlockRef>>()
            );
        }

        if let Some(last_commit) = last_commit {
            let mut index = last_commit.index();
            let gc_round = state.gc_round();
            info!(
                "Recovering block commit statuses from commit index {} and backwards until leader of round <= gc_round {:?}",
                index, gc_round
            );

            loop {
                let commits = store
                    .scan_commits((index..=index).into())
                    .unwrap_or_else(|e| panic!("Failed to read from storage: {:?}", e));
                let Some(commit) = commits.first() else {
                    info!("Recovering finished up to index {index}, no more commits to recover");
                    break;
                };

                // Check the commit leader round to see if it is within the gc_round. If it is not then we can stop the recovery process.
                if gc_round > 0 && commit.leader().round <= gc_round {
                    info!(
                        "Recovering finished, reached commit leader round {} <= gc_round {}",
                        commit.leader().round,
                        gc_round
                    );
                    break;
                }

                commit.blocks().iter().filter(|b| b.round > gc_round).for_each(|block_ref|{
                    debug!(
                        "Setting block {:?} as committed based on commit {:?}",
                        block_ref,
                        commit.index()
                    );
                    assert!(state.set_committed(block_ref), "Attempted to set again a block {:?} as committed when recovering commit {:?}", block_ref, commit);
                });

                // All commits are indexed starting from 1, so one reach zero exit.
                index = index.saturating_sub(1);
                if index == 0 {
                    break;
                }
            }
        }

        // Recover hard linked statuses for blocks within GC round.
        let proposed_blocks = store
            .scan_blocks_by_author(context.own_index, state.gc_round() + 1)
            .expect("Database error");
        for block in proposed_blocks {
            state.link_causal_history(block.reference());
        }

        state
    }

    /// Accepts a block into DagState and keeps it in memory.
    pub(crate) fn accept_block(&mut self, block: VerifiedBlock) {
        assert_ne!(block.round(), 0, "Genesis block should not be accepted into DAG.");

        let block_ref = block.reference();
        if self.contains_block(&block_ref) {
            return;
        }

        let now = self.context.clock.timestamp_utc_ms();
        if block.timestamp_ms() > now {
            trace!(
                "Block {:?} with timestamp {} is greater than local timestamp {}.",
                block,
                block.timestamp_ms(),
                now,
            );
        }

        // TODO: Move this check to core
        // Ensure we don't write multiple blocks per slot for our own index
        if block_ref.author == self.context.own_index {
            let existing_blocks = self.get_uncommitted_blocks_at_slot(block_ref.into());
            assert!(
                existing_blocks.is_empty(),
                "Block Rejected! Attempted to add block {block:#?} to own slot where \
                block(s) {existing_blocks:#?} already exists."
            );
        }
        self.update_block_metadata(&block);
        self.blocks_to_write.push(block);
        let source = if self.context.own_index == block_ref.author { "own" } else { "others" };
    }

    /// Updates internal metadata for a block.
    fn update_block_metadata(&mut self, block: &VerifiedBlock) {
        let block_ref = block.reference();
        self.recent_blocks.insert(block_ref, BlockInfo::new(block.clone()));
        self.recent_refs_by_authority[block_ref.author].insert(block_ref);

        if self.threshold_clock.add_block(block_ref) {
            // Do not measure quorum delay when no local block is proposed in the round.
            let last_proposed_block = self.get_last_proposed_block();
            if last_proposed_block.round() == block_ref.round {
                let quorum_delay_ms = self
                    .context
                    .clock
                    .timestamp_utc_ms()
                    .saturating_sub(self.get_last_proposed_block().timestamp_ms());
            }
        }

        self.highest_accepted_round = max(self.highest_accepted_round, block.round());

        let highest_accepted_round_for_author = self.recent_refs_by_authority[block_ref.author]
            .last()
            .map(|block_ref| block_ref.round)
            .expect("There should be by now at least one block ref");
    }

    /// Accepts a blocks into DagState and keeps it in memory.
    pub(crate) fn accept_blocks(&mut self, blocks: Vec<VerifiedBlock>) {
        debug!("Accepting blocks: {}", blocks.iter().map(|b| b.reference().to_string()).join(","));
        for block in blocks {
            self.accept_block(block);
        }
    }

    /// Gets a block by checking cached recent blocks then storage.
    /// Returns None when the block is not found.
    pub(crate) fn get_block(&self, reference: &BlockRef) -> Option<VerifiedBlock> {
        self.get_blocks(&[*reference]).pop().expect("Exactly one element should be returned")
    }

    /// Gets blocks by checking genesis, cached recent blocks in memory, then storage.
    /// An element is None when the corresponding block is not found.
    pub(crate) fn get_blocks(&self, block_refs: &[BlockRef]) -> Vec<Option<VerifiedBlock>> {
        let mut blocks = vec![None; block_refs.len()];
        let mut missing = Vec::new();

        for (index, block_ref) in block_refs.iter().enumerate() {
            if block_ref.round == GENESIS_ROUND {
                // Allow the caller to handle the invalid genesis ancestor error.
                if let Some(block) = self.genesis.get(block_ref) {
                    blocks[index] = Some(block.clone());
                }
                continue;
            }
            if let Some(block_info) = self.recent_blocks.get(block_ref) {
                blocks[index] = Some(block_info.block.clone());
                continue;
            }
            missing.push((index, block_ref));
        }

        if missing.is_empty() {
            return blocks;
        }

        let missing_refs = missing.iter().map(|(_, block_ref)| **block_ref).collect::<Vec<_>>();
        let store_results = self
            .store
            .read_blocks(&missing_refs)
            .unwrap_or_else(|e| panic!("Failed to read from storage: {:?}", e));

        for ((index, _), result) in missing.into_iter().zip(store_results.into_iter()) {
            blocks[index] = result;
        }

        blocks
    }

    /// Gets all uncommitted blocks in a slot.
    /// Uncommitted blocks must exist in memory, so only in-memory blocks are checked.
    pub(crate) fn get_uncommitted_blocks_at_slot(&self, slot: Slot) -> Vec<VerifiedBlock> {
        // TODO: either panic below when the slot is at or below the last committed round,
        // or support reading from storage while limiting storage reads to edge cases.

        let mut blocks = vec![];
        for (_block_ref, block_info) in self.recent_blocks.range((
            Included(BlockRef::new(slot.round, slot.authority, BlockDigest::MIN)),
            Included(BlockRef::new(slot.round, slot.authority, BlockDigest::MAX)),
        )) {
            blocks.push(block_info.block.clone())
        }
        blocks
    }

    /// Gets all uncommitted blocks in a round.
    /// Uncommitted blocks must exist in memory, so only in-memory blocks are checked.
    pub(crate) fn get_uncommitted_blocks_at_round(&self, round: Round) -> Vec<VerifiedBlock> {
        if round <= self.last_commit_round() {
            panic!("Round {} have committed blocks!", round);
        }

        let mut blocks = vec![];
        for (_block_ref, block_info) in self.recent_blocks.range((
            Included(BlockRef::new(round, AuthorityIndex::ZERO, BlockDigest::MIN)),
            Excluded(BlockRef::new(round + 1, AuthorityIndex::ZERO, BlockDigest::MIN)),
        )) {
            blocks.push(block_info.block.clone())
        }
        blocks
    }

    /// Gets all ancestors in the history of a block at a certain round.
    pub(crate) fn ancestors_at_round(
        &self,
        later_block: &VerifiedBlock,
        earlier_round: Round,
    ) -> Vec<VerifiedBlock> {
        // Iterate through ancestors of later_block in round descending order.
        let mut linked: BTreeSet<BlockRef> = later_block.ancestors().iter().cloned().collect();
        while !linked.is_empty() {
            let round = linked.last().unwrap().round;
            // Stop after finishing traversal for ancestors above earlier_round.
            if round <= earlier_round {
                break;
            }
            let block_ref = linked.pop_last().unwrap();
            let Some(block) = self.get_block(&block_ref) else {
                panic!("Block {:?} should exist in DAG!", block_ref);
            };
            linked.extend(block.ancestors().iter().cloned());
        }
        linked
            .range((
                Included(BlockRef::new(earlier_round, AuthorityIndex::ZERO, BlockDigest::MIN)),
                Unbounded,
            ))
            .map(|r| {
                self.get_block(r)
                    .unwrap_or_else(|| panic!("Block {:?} should exist in DAG!", r))
                    .clone()
            })
            .collect()
    }

    /// Gets the last proposed block from this authority.
    /// If no block is proposed yet, returns the genesis block.
    pub(crate) fn get_last_proposed_block(&self) -> VerifiedBlock {
        self.get_last_block_for_authority(self.context.own_index)
    }

    /// Retrieves the last accepted block from the specified `authority`. If no block is found in cache
    /// then the genesis block is returned as no other block has been received from that authority.
    pub(crate) fn get_last_block_for_authority(&self, authority: AuthorityIndex) -> VerifiedBlock {
        if let Some(last) = self.recent_refs_by_authority[authority].last() {
            return self
                .recent_blocks
                .get(last)
                .expect("Block should be found in recent blocks")
                .block
                .clone();
        }

        // if none exists, then fallback to genesis
        let (_, genesis_block) = self
            .genesis
            .iter()
            .find(|(block_ref, _)| block_ref.author == authority)
            .expect("Genesis should be found for authority {authority_index}");
        genesis_block.clone()
    }

    /// Returns cached recent blocks from the specified authority.
    /// Blocks returned are limited to round >= `start`, and cached.
    /// NOTE: caller should not assume returned blocks are always chained.
    /// "Disconnected" blocks can be returned when there are byzantine blocks,
    /// or a previously evicted block is accepted again.
    pub(crate) fn get_cached_blocks(
        &self,
        authority: AuthorityIndex,
        start: Round,
    ) -> Vec<VerifiedBlock> {
        self.get_cached_blocks_in_range(authority, start, Round::MAX, usize::MAX)
    }

    // Retrieves the cached block within the range [start_round, end_round) from a given authority,
    // limited in total number of blocks.
    pub(crate) fn get_cached_blocks_in_range(
        &self,
        authority: AuthorityIndex,
        start_round: Round,
        end_round: Round,
        limit: usize,
    ) -> Vec<VerifiedBlock> {
        if start_round >= end_round || limit == 0 {
            return vec![];
        }

        let mut blocks = vec![];
        for block_ref in self.recent_refs_by_authority[authority].range((
            Included(BlockRef::new(start_round, authority, BlockDigest::MIN)),
            Excluded(BlockRef::new(end_round, AuthorityIndex::MIN, BlockDigest::MIN)),
        )) {
            let block_info =
                self.recent_blocks.get(block_ref).expect("Block should exist in recent blocks");
            blocks.push(block_info.block.clone());
            if blocks.len() >= limit {
                break;
            }
        }
        blocks
    }

    // Retrieves the last cached block within the range [start_round, end_round) from a given authority.
    pub(crate) fn get_last_cached_block_in_range(
        &self,
        authority: AuthorityIndex,
        start_round: Round,
        end_round: Round,
    ) -> Option<VerifiedBlock> {
        if start_round >= end_round {
            return None;
        }

        let block_ref = self.recent_refs_by_authority[authority]
            .range((
                Included(BlockRef::new(start_round, authority, BlockDigest::MIN)),
                Excluded(BlockRef::new(end_round, AuthorityIndex::MIN, BlockDigest::MIN)),
            ))
            .last()?;

        self.recent_blocks.get(block_ref).map(|block_info| block_info.block.clone())
    }

    /// Returns the last block proposed per authority with `evicted round < round < end_round`.
    /// The method is guaranteed to return results only when the `end_round` is not earlier of the
    /// available cached data for each authority (evicted round + 1), otherwise the method will panic.
    /// It's the caller's responsibility to ensure that is not requesting for earlier rounds.
    /// In case of equivocation for an authority's last slot, one block will be returned (the last in order)
    /// and the other equivocating blocks will be returned.
    pub(crate) fn get_last_cached_block_per_authority(
        &self,
        end_round: Round,
    ) -> Vec<(VerifiedBlock, Vec<BlockRef>)> {
        // Initialize with the genesis blocks as fallback
        let mut blocks = self.genesis.values().cloned().collect::<Vec<_>>();
        let mut equivocating_blocks = vec![vec![]; self.context.committee.size()];

        if end_round == GENESIS_ROUND {
            panic!(
                "Attempted to retrieve blocks earlier than the genesis round which is not possible"
            );
        }

        if end_round == GENESIS_ROUND + 1 {
            return blocks.into_iter().map(|b| (b, vec![])).collect();
        }

        for (authority_index, block_refs) in self.recent_refs_by_authority.iter().enumerate() {
            let authority_index =
                self.context.committee.to_authority_index(authority_index).unwrap();

            let last_evicted_round = self.evicted_rounds[authority_index];
            if end_round.saturating_sub(1) <= last_evicted_round {
                panic!(
                    "Attempted to request for blocks of rounds < {end_round}, when the last evicted round is {last_evicted_round} for authority {authority_index}",
                );
            }

            let block_ref_iter = block_refs
                .range((
                    Included(BlockRef::new(
                        last_evicted_round + 1,
                        authority_index,
                        BlockDigest::MIN,
                    )),
                    Excluded(BlockRef::new(end_round, authority_index, BlockDigest::MIN)),
                ))
                .rev();

            let mut last_round = 0;
            for block_ref in block_ref_iter {
                if last_round == 0 {
                    last_round = block_ref.round;
                    let block_info = self
                        .recent_blocks
                        .get(block_ref)
                        .expect("Block should exist in recent blocks");
                    blocks[authority_index] = block_info.block.clone();
                    continue;
                }
                if block_ref.round < last_round {
                    break;
                }
                equivocating_blocks[authority_index].push(*block_ref);
            }
        }

        blocks.into_iter().zip(equivocating_blocks).collect()
    }

    /// Checks whether a block exists in the slot. The method checks only against the cached data.
    /// If the user asks for a slot that is not within the cached data then a panic is thrown.
    pub(crate) fn contains_cached_block_at_slot(&self, slot: Slot) -> bool {
        // Always return true for genesis slots.
        if slot.round == GENESIS_ROUND {
            return true;
        }

        let eviction_round = self.evicted_rounds[slot.authority];
        if slot.round <= eviction_round {
            panic!(
                "{}",
                format!(
                    "Attempted to check for slot {slot} that is <= the last evicted round {eviction_round}"
                )
            );
        }

        let mut result = self.recent_refs_by_authority[slot.authority].range((
            Included(BlockRef::new(slot.round, slot.authority, BlockDigest::MIN)),
            Included(BlockRef::new(slot.round, slot.authority, BlockDigest::MAX)),
        ));
        result.next().is_some()
    }

    /// Checks whether the required blocks are in cache, if exist, or otherwise will check in store. The method is not caching
    /// back the results, so its expensive if keep asking for cache missing blocks.
    pub(crate) fn contains_blocks(&self, block_refs: Vec<BlockRef>) -> Vec<bool> {
        let mut exist = vec![false; block_refs.len()];
        let mut missing = Vec::new();

        for (index, block_ref) in block_refs.into_iter().enumerate() {
            let recent_refs = &self.recent_refs_by_authority[block_ref.author];
            if recent_refs.contains(&block_ref) || self.genesis.contains_key(&block_ref) {
                exist[index] = true;
            } else if recent_refs.is_empty() || recent_refs.last().unwrap().round < block_ref.round
            {
                // Optimization: recent_refs contain the most recent blocks known to this authority.
                // If a block ref is not found there and has a higher round, it definitely is
                // missing from this authority and there is no need to check disk.
                exist[index] = false;
            } else {
                missing.push((index, block_ref));
            }
        }

        if missing.is_empty() {
            return exist;
        }

        let missing_refs = missing.iter().map(|(_, block_ref)| *block_ref).collect::<Vec<_>>();
        let store_results = self
            .store
            .contains_blocks(&missing_refs)
            .unwrap_or_else(|e| panic!("Failed to read from storage: {:?}", e));

        for ((index, _), result) in missing.into_iter().zip(store_results.into_iter()) {
            exist[index] = result;
        }

        exist
    }

    pub(crate) fn contains_block(&self, block_ref: &BlockRef) -> bool {
        let blocks = self.contains_blocks(vec![*block_ref]);
        blocks.first().cloned().unwrap()
    }

    // Sets the block as committed in the cache. If the block is set as committed for first time, then true is returned, otherwise false is returned instead.
    // Method will panic if the block is not found in the cache.
    pub(crate) fn set_committed(&mut self, block_ref: &BlockRef) -> bool {
        if let Some(block_info) = self.recent_blocks.get_mut(block_ref) {
            if !block_info.committed {
                block_info.committed = true;
                return true;
            }
            false
        } else {
            panic!("Block {:?} not found in cache to set as committed.", block_ref);
        }
    }

    /// Returns true if the block is committed. Only valid for blocks above the GC round.
    pub(crate) fn is_committed(&self, block_ref: &BlockRef) -> bool {
        self.recent_blocks
            .get(block_ref)
            .unwrap_or_else(|| panic!("Attempted to query for commit status for a block not in cached data {block_ref}"))
            .committed
    }

    /// Recursively sets blocks in the causal history of the root block as hard linked, including the root block itself.
    /// Returns the list of blocks that are newly linked.
    /// The returned blocks are guaranteed to be above the GC round.
    pub(crate) fn link_causal_history(&mut self, root_block: BlockRef) -> Vec<BlockRef> {
        let gc_round = self.gc_round();
        let mut linked_blocks = vec![];
        let mut targets = VecDeque::new();
        targets.push_back(root_block);
        while let Some(block_ref) = targets.pop_front() {
            // This is only correct with GC enabled.
            if block_ref.round <= gc_round {
                continue;
            }
            let block_info = self
                .recent_blocks
                .get_mut(&block_ref)
                .unwrap_or_else(|| panic!("Block {:?} is not in DAG state", block_ref));
            if block_info.included {
                continue;
            }
            linked_blocks.push(block_ref);
            block_info.included = true;
            targets.extend(block_info.block.ancestors().iter());
        }
        linked_blocks
    }

    /// Returns true if the block has been included in an owned proposed block.
    /// NOTE: caller should make sure only blocks above GC round are queried.
    pub(crate) fn has_been_included(&self, block_ref: &BlockRef) -> bool {
        self.recent_blocks
            .get(block_ref)
            .unwrap_or_else(|| {
                panic!(
                    "Attempted to query for inclusion status for a block not in cached data {}",
                    block_ref
                )
            })
            .included
    }

    pub(crate) fn threshold_clock_round(&self) -> Round {
        self.threshold_clock.get_round()
    }

    // The timestamp of when quorum threshold was last reached in the threshold clock.
    pub(crate) fn threshold_clock_quorum_ts(&self) -> Instant {
        self.threshold_clock.get_quorum_ts()
    }

    pub(crate) fn highest_accepted_round(&self) -> Round {
        self.highest_accepted_round
    }

    // Buffers a new commit in memory and updates last committed rounds.
    // REQUIRED: must not skip over any commit index.
    pub(crate) fn add_commit(&mut self, commit: TrustedCommit) {
        let time_diff = if let Some(last_commit) = &self.last_commit {
            if commit.index() <= last_commit.index() {
                error!(
                    "New commit index {} <= last commit index {}!",
                    commit.index(),
                    last_commit.index()
                );
                return;
            }
            assert_eq!(commit.index(), last_commit.index() + 1);

            if commit.timestamp_ms() < last_commit.timestamp_ms() {
                panic!(
                    "Commit timestamps do not monotonically increment, prev commit {:?}, new commit {:?}",
                    last_commit, commit
                );
            }
            commit.timestamp_ms().saturating_sub(last_commit.timestamp_ms())
        } else {
            assert_eq!(commit.index(), 1);
            0
        };

        let commit_round_advanced = if let Some(previous_commit) = &self.last_commit {
            previous_commit.round() < commit.round()
        } else {
            true
        };

        self.last_commit = Some(commit.clone());

        if commit_round_advanced {
            let now = std::time::Instant::now();
            if let Some(previous_time) = self.last_commit_round_advancement_time {}
            self.last_commit_round_advancement_time = Some(now);
        }

        for block_ref in commit.blocks().iter() {
            self.last_committed_rounds[block_ref.author] =
                max(self.last_committed_rounds[block_ref.author], block_ref.round);
        }

        for (i, round) in self.last_committed_rounds.iter().enumerate() {
            let index = self.context.committee.to_authority_index(i).unwrap();
        }

        self.pending_commit_votes.push_back(commit.reference());
        self.commits_to_write.push(commit);
    }

    /// Recovers commits to write from storage, at startup.
    pub(crate) fn recover_commits_to_write(&mut self, commits: Vec<TrustedCommit>) {
        self.commits_to_write.extend(commits);
    }

    pub(crate) fn ensure_commits_to_write_is_empty(&self) {
        assert!(
            self.commits_to_write.is_empty(),
            "Commits to write should be empty. {:?}",
            self.commits_to_write,
        );
    }

    pub(crate) fn add_commit_info(&mut self, reputation_scores: ReputationScores) {
        // We create an empty scoring subdag once reputation scores are calculated.
        // Note: It is okay for this to not be gated by protocol config as the
        // scoring_subdag should be empty in either case at this point.
        assert!(self.scoring_subdag.is_empty());

        let commit_info =
            CommitInfo { committed_rounds: self.last_committed_rounds.clone(), reputation_scores };
        let last_commit = self.last_commit.as_ref().expect("Last commit should already be set.");
        self.commit_info_to_write.push((last_commit.reference(), commit_info));
    }

    pub(crate) fn add_finalized_commit(
        &mut self,
        commit_ref: CommitRef,
        rejected_transactions: BTreeMap<BlockRef, Vec<TransactionIndex>>,
    ) {
        self.finalized_commits_to_write.push((commit_ref, rejected_transactions));
    }

    pub(crate) fn take_commit_votes(&mut self, limit: usize) -> Vec<CommitVote> {
        let mut votes = Vec::new();
        while !self.pending_commit_votes.is_empty() && votes.len() < limit {
            votes.push(self.pending_commit_votes.pop_front().unwrap());
        }
        votes
    }

    /// Index of the last commit.
    pub(crate) fn last_commit_index(&self) -> CommitIndex {
        match &self.last_commit {
            Some(commit) => commit.index(),
            None => 0,
        }
    }

    /// Digest of the last commit.
    pub(crate) fn last_commit_digest(&self) -> CommitDigest {
        match &self.last_commit {
            Some(commit) => commit.digest(),
            None => CommitDigest::MIN,
        }
    }

    /// Timestamp of the last commit.
    pub(crate) fn last_commit_timestamp_ms(&self) -> BlockTimestampMs {
        match &self.last_commit {
            Some(commit) => commit.timestamp_ms(),
            None => 0,
        }
    }

    /// Leader slot of the last commit.
    pub(crate) fn last_commit_leader(&self) -> Slot {
        match &self.last_commit {
            Some(commit) => commit.leader().into(),
            None => self
                .genesis
                .iter()
                .next()
                .map(|(genesis_ref, _)| *genesis_ref)
                .expect("Genesis blocks should always be available.")
                .into(),
        }
    }

    /// Highest round where a block is committed, which is last commit's leader round.
    pub(crate) fn last_commit_round(&self) -> Round {
        match &self.last_commit {
            Some(commit) => commit.leader().round,
            None => 0,
        }
    }

    /// Last committed round per authority.
    pub(crate) fn last_committed_rounds(&self) -> Vec<Round> {
        self.last_committed_rounds.clone()
    }

    /// The GC round is the highest round that blocks of equal or lower round are considered obsolete and no longer possible to be committed.
    /// There is no meaning accepting any blocks with round <= gc_round. The Garbage Collection (GC) round is calculated based on the latest
    /// committed leader round. When GC is disabled that will return the genesis round.
    pub(crate) fn gc_round(&self) -> Round {
        self.calculate_gc_round(self.last_commit_round())
    }

    /// Calculates the GC round from the input leader round, which can be different
    /// from the last committed leader round.
    pub(crate) fn calculate_gc_round(&self, commit_round: Round) -> Round {
        commit_round.saturating_sub(self.context.protocol_config.gc_depth())
    }

    /// Flushes unpersisted blocks, commits and commit info to storage.
    ///
    /// REQUIRED: when buffering a block, all of its ancestors and the latest commit which sets the GC round
    /// must also be buffered.
    /// REQUIRED: when buffering a commit, all of its included blocks and the previous commits must also be buffered.
    /// REQUIRED: when flushing, all of the buffered blocks and commits must be flushed together to ensure consistency.
    ///
    /// After each flush, DagState becomes persisted in storage and it expected to recover
    /// all internal states from storage after restarts.
    pub(crate) fn flush(&mut self) {
        // Flush buffered data to storage.
        let pending_blocks = std::mem::take(&mut self.blocks_to_write);
        let pending_commits = std::mem::take(&mut self.commits_to_write);
        let pending_commit_info = std::mem::take(&mut self.commit_info_to_write);
        let pending_finalized_commits = std::mem::take(&mut self.finalized_commits_to_write);
        if pending_blocks.is_empty()
            && pending_commits.is_empty()
            && pending_commit_info.is_empty()
            && pending_finalized_commits.is_empty()
        {
            return;
        }

        debug!(
            "Flushing {} blocks ({}), {} commits ({}), {} commit infos ({}), {} finalized commits ({}) to storage.",
            pending_blocks.len(),
            pending_blocks.iter().map(|b| b.reference().to_string()).join(","),
            pending_commits.len(),
            pending_commits.iter().map(|c| c.reference().to_string()).join(","),
            pending_commit_info.len(),
            pending_commit_info.iter().map(|(commit_ref, _)| commit_ref.to_string()).join(","),
            pending_finalized_commits.len(),
            pending_finalized_commits
                .iter()
                .map(|(commit_ref, _)| commit_ref.to_string())
                .join(","),
        );
        self.store
            .write(WriteBatch::new(
                pending_blocks,
                pending_commits,
                pending_commit_info,
                pending_finalized_commits,
            ))
            .unwrap_or_else(|e| panic!("Failed to write to storage: {:?}", e));

        // Clean up old cached data. After flushing, all cached blocks are guaranteed to be persisted.
        for (authority_index, _) in self.context.committee.authorities() {
            let eviction_round = self.calculate_authority_eviction_round(authority_index);
            while let Some(block_ref) = self.recent_refs_by_authority[authority_index].first() {
                if block_ref.round <= eviction_round {
                    self.recent_blocks.remove(block_ref);
                    self.recent_refs_by_authority[authority_index].pop_first();
                } else {
                    break;
                }
            }
            self.evicted_rounds[authority_index] = eviction_round;
        }
    }

    pub(crate) fn recover_last_commit_info(&self) -> Option<(CommitRef, CommitInfo)> {
        self.store
            .read_last_commit_info()
            .unwrap_or_else(|e| panic!("Failed to read from storage: {:?}", e))
    }

    pub(crate) fn add_scoring_subdags(&mut self, scoring_subdags: Vec<CommittedSubDag>) {
        self.scoring_subdag.add_subdags(scoring_subdags);
    }

    pub(crate) fn clear_scoring_subdag(&mut self) {
        self.scoring_subdag.clear();
    }

    pub(crate) fn scoring_subdags_count(&self) -> usize {
        self.scoring_subdag.scored_subdags_count()
    }

    pub(crate) fn is_scoring_subdag_empty(&self) -> bool {
        self.scoring_subdag.is_empty()
    }

    pub(crate) fn calculate_scoring_subdag_scores(&self) -> ReputationScores {
        self.scoring_subdag.calculate_distributed_vote_scores()
    }

    pub(crate) fn scoring_subdag_commit_range(&self) -> CommitIndex {
        self.scoring_subdag
            .commit_range
            .as_ref()
            .expect("commit range should exist for scoring subdag")
            .end()
    }

    /// The last round that should get evicted after a cache clean up operation. After this round we are
    /// guaranteed to have all the produced blocks from that authority. For any round that is
    /// <= `last_evicted_round` we don't have such guarantees as out of order blocks might exist.
    fn calculate_authority_eviction_round(&self, authority_index: AuthorityIndex) -> Round {
        let last_round = self.recent_refs_by_authority[authority_index]
            .last()
            .map(|block_ref| block_ref.round)
            .unwrap_or(GENESIS_ROUND);

        Self::eviction_round(last_round, self.gc_round(), self.cached_rounds)
    }

    /// Calculates the eviction round for the given authority. The goal is to keep at least `cached_rounds`
    /// of the latest blocks in the cache (if enough data is available), while evicting blocks with rounds <= `gc_round` when possible.
    fn eviction_round(last_round: Round, gc_round: Round, cached_rounds: u32) -> Round {
        gc_round.min(last_round.saturating_sub(cached_rounds))
    }

    /// Returns the underlying store.
    pub(crate) fn store(&self) -> Arc<dyn Store> {
        self.store.clone()
    }

    /// Detects and returns the blocks of the round that forms the last quorum. The method will return
    /// the quorum even if that's genesis.
    #[cfg(test)]
    pub(crate) fn last_quorum(&self) -> Vec<VerifiedBlock> {
        // the quorum should exist either on the highest accepted round or the one before. If we fail to detect
        // a quorum then it means that our DAG has advanced with missing causal history.
        for round in
            (self.highest_accepted_round.saturating_sub(1)..=self.highest_accepted_round).rev()
        {
            if round == GENESIS_ROUND {
                return self.genesis_blocks();
            }
            use types::consensus::stake_aggregator::{QuorumThreshold, StakeAggregator};
            let mut quorum = StakeAggregator::<QuorumThreshold>::new();

            // Since the minimum wave length is 3 we expect to find a quorum in the uncommitted rounds.
            let blocks = self.get_uncommitted_blocks_at_round(round);
            for block in &blocks {
                if quorum.add(block.author(), &self.context.committee) {
                    return blocks;
                }
            }
        }

        panic!("Fatal error, no quorum has been detected in our DAG on the last two rounds.");
    }

    #[cfg(test)]
    pub(crate) fn genesis_blocks(&self) -> Vec<VerifiedBlock> {
        self.genesis.values().cloned().collect()
    }

    #[cfg(test)]
    pub(crate) fn set_last_commit(&mut self, commit: TrustedCommit) {
        self.last_commit = Some(commit);
    }
}

struct BlockInfo {
    block: VerifiedBlock,
    // Whether the block has been committed
    committed: bool,
    // Whether the block has been included in the causal history of an owned proposed block.
    ///
    /// There are two usages of this field:
    /// 1. When proposing blocks, determine the set of blocks to carry votes for.
    /// 2. When recovering, determine if a block has not been included in a proposed block and
    ///    should recover transaction votes by voting.
    included: bool,
}

impl BlockInfo {
    fn new(block: VerifiedBlock) -> Self {
        Self { block, committed: false, included: false }
    }
}
