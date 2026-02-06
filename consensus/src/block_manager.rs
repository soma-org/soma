use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

use itertools::Itertools as _;

use crate::dag_state::DagState;
use parking_lot::RwLock;
use tracing::{debug, trace, warn};
use types::consensus::{
    block::{BlockAPI, BlockRef, GENESIS_ROUND, Round, VerifiedBlock},
    context::Context,
};

struct SuspendedBlock {
    block: VerifiedBlock,
    missing_ancestors: BTreeSet<BlockRef>,
    timestamp: Instant,
}

impl SuspendedBlock {
    fn new(block: VerifiedBlock, missing_ancestors: BTreeSet<BlockRef>) -> Self {
        Self { block, missing_ancestors, timestamp: Instant::now() }
    }
}

/// Block manager suspends incoming blocks until they are connected to the existing graph,
/// returning newly connected blocks.
/// TODO: As it is possible to have Byzantine validators who produce Blocks without valid causal
/// history we need to make sure that BlockManager takes care of that and avoid OOM (Out Of Memory)
/// situations.
pub(crate) struct BlockManager {
    context: Arc<Context>,
    dag_state: Arc<RwLock<DagState>>,

    /// Keeps all the suspended blocks. A suspended block is a block that is missing part of its causal history and thus
    /// can't be immediately processed. A block will remain in this map until all its causal history has been successfully
    /// processed.
    suspended_blocks: BTreeMap<BlockRef, SuspendedBlock>,
    /// A map that keeps all the blocks that we are missing (keys) and the corresponding blocks that reference the missing blocks
    /// as ancestors and need them to get unsuspended. It is possible for a missing dependency (key) to be a suspended block, so
    /// the block has been already fetched but it self is still missing some of its ancestors to be processed.
    missing_ancestors: BTreeMap<BlockRef, BTreeSet<BlockRef>>,
    /// Keeps all the blocks that we actually miss and haven't fetched them yet. That set will basically contain all the
    /// keys from the `missing_ancestors` minus any keys that exist in `suspended_blocks`.
    missing_blocks: BTreeSet<BlockRef>,
    /// A vector that holds a tuple of (lowest_round, highest_round) of received blocks per authority.
    /// This is used for metrics reporting purposes and resets during restarts.
    received_block_rounds: Vec<Option<(Round, Round)>>,
}

impl BlockManager {
    pub(crate) fn new(context: Arc<Context>, dag_state: Arc<RwLock<DagState>>) -> Self {
        let committee_size = context.committee.size();
        Self {
            context,
            dag_state,
            suspended_blocks: BTreeMap::new(),
            missing_ancestors: BTreeMap::new(),
            missing_blocks: BTreeSet::new(),
            received_block_rounds: vec![None; committee_size],
        }
    }

    /// Tries to accept the provided blocks assuming that all their causal history exists. The method
    /// returns all the blocks that have been successfully processed in round ascending order, that includes also previously
    /// suspended blocks that have now been able to get accepted. Method also returns a set with the missing ancestor blocks.
    #[tracing::instrument(skip_all)]
    pub(crate) fn try_accept_blocks(
        &mut self,
        blocks: Vec<VerifiedBlock>,
    ) -> (Vec<VerifiedBlock>, BTreeSet<BlockRef>) {
        self.try_accept_blocks_internal(blocks, false)
    }

    // Tries to accept blocks that have been committed. Returns all the blocks that have been accepted, both from the ones
    // provided and any children blocks.
    #[tracing::instrument(skip_all)]
    pub(crate) fn try_accept_committed_blocks(
        &mut self,
        blocks: Vec<VerifiedBlock>,
    ) -> Vec<VerifiedBlock> {
        // Just accept the blocks

        let (accepted_blocks, missing_blocks) = self.try_accept_blocks_internal(blocks, true);
        assert!(
            missing_blocks.is_empty(),
            "No missing blocks should be returned for committed blocks"
        );

        accepted_blocks
    }

    /// Attempts to accept the provided blocks. When `committed = true` then the blocks are considered to be committed via certified commits and
    /// are handled differently.
    fn try_accept_blocks_internal(
        &mut self,
        mut blocks: Vec<VerifiedBlock>,
        committed: bool,
    ) -> (Vec<VerifiedBlock>, BTreeSet<BlockRef>) {
        blocks.sort_by_key(|b| b.round());
        if !blocks.is_empty() {
            debug!(
                "Trying to accept blocks: {}",
                blocks.iter().map(|b| b.reference().to_string()).join(",")
            );
        }

        let mut accepted_blocks = vec![];
        let mut missing_blocks = BTreeSet::new();

        for block in blocks {
            self.update_block_received_metrics(&block);

            // Try to accept the input block.
            let block_ref = block.reference();

            let mut blocks_to_accept = vec![];
            if committed {
                match self.try_accept_one_committed_block(block) {
                    TryAcceptResult::Accepted(block) => {
                        // As this is a committed block, then it's already accepted and there is no need to verify its timestamps.
                        // Just add it to the accepted blocks list.
                        accepted_blocks.push(block);
                    }
                    TryAcceptResult::Processed => continue,
                    TryAcceptResult::Suspended(_) | TryAcceptResult::Skipped => panic!(
                        "Did not expect to suspend or skip a committed block: {:?}",
                        block_ref
                    ),
                };
            } else {
                match self.try_accept_one_block(block) {
                    TryAcceptResult::Accepted(block) => {
                        blocks_to_accept.push(block);
                    }
                    TryAcceptResult::Suspended(ancestors_to_fetch) => {
                        debug!(
                            "Missing ancestors to fetch for block {block_ref}: {}",
                            ancestors_to_fetch.iter().map(|b| b.to_string()).join(",")
                        );
                        missing_blocks.extend(ancestors_to_fetch);
                        continue;
                    }
                    TryAcceptResult::Processed | TryAcceptResult::Skipped => continue,
                };
            };

            // If the block is accepted, try to unsuspend its children blocks if any.
            let unsuspended_blocks = self.try_unsuspend_children_blocks(block_ref);
            blocks_to_accept.extend(unsuspended_blocks);

            // Insert the accepted blocks into DAG state so future blocks including them as
            // ancestors do not get suspended.
            self.dag_state.write().accept_blocks(blocks_to_accept.clone());

            accepted_blocks.extend(blocks_to_accept);
        }

        // Figure out the new missing blocks
        (accepted_blocks, missing_blocks)
    }

    fn try_accept_one_committed_block(&mut self, block: VerifiedBlock) -> TryAcceptResult {
        if self.dag_state.read().contains_block(&block.reference()) {
            return TryAcceptResult::Processed;
        }

        // Remove the block from missing and suspended blocks
        self.missing_blocks.remove(&block.reference());

        // If the block has been already fetched and parked as suspended block, then remove it. Also find all the references of missing
        // ancestors to remove those as well. If we don't do that then it's possible once the missing ancestor is fetched to cause a panic
        // when trying to unsuspend this children as it won't be found in the suspended blocks map.
        if let Some(suspended_block) = self.suspended_blocks.remove(&block.reference()) {
            suspended_block.missing_ancestors.iter().for_each(|ancestor| {
                if let Some(references) = self.missing_ancestors.get_mut(ancestor) {
                    references.remove(&block.reference());
                }
            });
        }

        // Accept this block before any unsuspended children blocks
        self.dag_state.write().accept_blocks(vec![block.clone()]);

        TryAcceptResult::Accepted(block)
    }

    /// Tries to find the provided block_refs in DagState and BlockManager,
    /// and returns missing block refs.
    pub(crate) fn try_find_blocks(&mut self, block_refs: Vec<BlockRef>) -> BTreeSet<BlockRef> {
        let gc_round = self.dag_state.read().gc_round();

        // No need to fetch blocks that are <= gc_round as they won't get processed anyways and they'll get skipped.
        // So keep only the ones above.
        let mut block_refs = block_refs
            .into_iter()
            .filter(|block_ref| block_ref.round > gc_round)
            .collect::<Vec<_>>();

        if block_refs.is_empty() {
            return BTreeSet::new();
        }

        block_refs.sort_by_key(|b| b.round);

        trace!("Trying to find blocks: {}", block_refs.iter().map(|b| b.to_string()).join(","));

        let mut missing_blocks = BTreeSet::new();

        for (found, block_ref) in self
            .dag_state
            .read()
            .contains_blocks(block_refs.clone())
            .into_iter()
            .zip(block_refs.iter())
        {
            if found || self.suspended_blocks.contains_key(block_ref) {
                continue;
            }
            // Fetches the block if it is not in dag state or suspended.
            missing_blocks.insert(*block_ref);
            if self.missing_blocks.insert(*block_ref) {
                // We want to report this as a missing ancestor even if there is no block that is actually references it right now. That will allow us
                // to seamlessly GC the block later if needed.
                self.missing_ancestors.entry(*block_ref).or_default();
            }
        }

        missing_blocks
    }

    /// Tries to accept the provided block. To accept a block its ancestors must have been already successfully accepted. If
    /// block is accepted then Some result is returned. None is returned when either the block is suspended or the block
    /// has been already accepted before.
    fn try_accept_one_block(&mut self, block: VerifiedBlock) -> TryAcceptResult {
        let block_ref = block.reference();
        let mut missing_ancestors = BTreeSet::new();
        let mut ancestors_to_fetch = BTreeSet::new();
        let dag_state = self.dag_state.read();
        let gc_round = dag_state.gc_round();

        // If block has been already received and suspended, or already processed and stored, or is a genesis block, then skip it.
        if self.suspended_blocks.contains_key(&block_ref) || dag_state.contains_block(&block_ref) {
            return TryAcceptResult::Processed;
        }

        // If the block is <= gc_round, then we simply skip its processing as there is no meaning do any action on it or even store it.
        if block.round() <= gc_round {
            return TryAcceptResult::Skipped;
        }

        // Keep only the ancestors that are greater than the GC round to check for their existence.
        let ancestors = block
            .ancestors()
            .iter()
            .filter(|ancestor| ancestor.round == GENESIS_ROUND || ancestor.round > gc_round)
            .cloned()
            .collect::<Vec<_>>();

        // make sure that we have all the required ancestors in store
        for (found, ancestor) in
            dag_state.contains_blocks(ancestors.clone()).into_iter().zip(ancestors.iter())
        {
            if !found {
                missing_ancestors.insert(*ancestor);

                // mark the block as having missing ancestors
                self.missing_ancestors.entry(*ancestor).or_default().insert(block_ref);

                // Add the ancestor to the missing blocks set only if it doesn't already exist in the suspended blocks - meaning
                // that we already have its payload.
                if !self.suspended_blocks.contains_key(ancestor) {
                    // Fetches the block if it is not in dag state or suspended.
                    ancestors_to_fetch.insert(*ancestor);
                    if self.missing_blocks.insert(*ancestor) {}
                }
            }
        }

        // Remove the block ref from the `missing_blocks` - if exists - since we now have received the block. The block
        // might still get suspended, but we won't report it as missing in order to not re-fetch.
        self.missing_blocks.remove(&block.reference());

        if !missing_ancestors.is_empty() {
            self.suspended_blocks.insert(block_ref, SuspendedBlock::new(block, missing_ancestors));
            return TryAcceptResult::Suspended(ancestors_to_fetch);
        }

        TryAcceptResult::Accepted(block)
    }

    /// Given an accepted block `accepted_block` it attempts to accept all the suspended children blocks assuming such exist.
    /// All the unsuspended / accepted blocks are returned as a vector in causal order.
    fn try_unsuspend_children_blocks(&mut self, accepted_block: BlockRef) -> Vec<VerifiedBlock> {
        let mut unsuspended_blocks = vec![];
        let mut to_process_blocks = vec![accepted_block];

        while let Some(block_ref) = to_process_blocks.pop() {
            // And try to check if its direct children can be unsuspended
            if let Some(block_refs_with_missing_deps) = self.missing_ancestors.remove(&block_ref) {
                for r in block_refs_with_missing_deps {
                    // For each dependency try to unsuspend it. If that's successful then we add it to the queue so
                    // we can recursively try to unsuspend its children.
                    if let Some(block) = self.try_unsuspend_block(&r, &block_ref) {
                        to_process_blocks.push(block.block.reference());
                        unsuspended_blocks.push(block);
                    }
                }
            }
        }

        let now = Instant::now();

        // Report the unsuspended blocks

        unsuspended_blocks.into_iter().map(|block| block.block).collect()
    }

    /// Attempts to unsuspend a block by checking its ancestors and removing the `accepted_dependency` by its local set.
    /// If there is no missing dependency then this block can be unsuspended immediately and is removed from the `suspended_blocks` map.
    fn try_unsuspend_block(
        &mut self,
        block_ref: &BlockRef,
        accepted_dependency: &BlockRef,
    ) -> Option<SuspendedBlock> {
        let block =
            self.suspended_blocks.get_mut(block_ref).expect("Block should be in suspended map");

        assert!(
            block.missing_ancestors.remove(accepted_dependency),
            "Block reference {} should be present in missing dependencies of {:?}",
            block_ref,
            block.block
        );

        if block.missing_ancestors.is_empty() {
            // we have no missing dependency, so we unsuspend the block and return it
            return self.suspended_blocks.remove(block_ref);
        }
        None
    }

    /// Tries to unsuspend any blocks for the latest gc round. If gc round hasn't changed then no blocks will be unsuspended due to
    /// this action.
    pub(crate) fn try_unsuspend_blocks_for_latest_gc_round(&mut self) {
        let gc_round = self.dag_state.read().gc_round();
        let mut blocks_unsuspended_below_gc_round = 0;
        let mut blocks_gc_ed = 0;

        while let Some((block_ref, _children_refs)) = self.missing_ancestors.first_key_value() {
            // If the first block in the missing ancestors is higher than the gc_round, then we can't unsuspend it yet. So we just put it back
            // and we terminate the iteration as any next entry will be of equal or higher round anyways.
            if block_ref.round > gc_round {
                return;
            }

            blocks_gc_ed += 1;

            assert!(
                !self.suspended_blocks.contains_key(block_ref),
                "Block should not be suspended, as we are causally GC'ing and no suspended block should exist for a missing ancestor."
            );

            // Also remove it from the missing list - we don't want to keep looking for it.
            self.missing_blocks.remove(block_ref);

            // Find all the children blocks that have a dependency on this one and try to unsuspend them
            let unsuspended_blocks = self.try_unsuspend_children_blocks(*block_ref);

            unsuspended_blocks.iter().for_each(|block| {
                if block.round() <= gc_round {
                    blocks_unsuspended_below_gc_round += 1;
                }
            });

            // Now accept the unsuspended blocks
            self.dag_state.write().accept_blocks(unsuspended_blocks.clone());
        }

        debug!(
            "Total {} blocks unsuspended and total blocks {} gc'ed <= gc_round {}",
            blocks_unsuspended_below_gc_round, blocks_gc_ed, gc_round
        );
    }

    /// Returns all the blocks that are currently missing and needed in order to accept suspended
    /// blocks.
    pub(crate) fn missing_blocks(&self) -> BTreeSet<BlockRef> {
        self.missing_blocks.clone()
    }

    fn update_block_received_metrics(&mut self, block: &VerifiedBlock) {
        let (min_round, max_round) =
            if let Some((curr_min, curr_max)) = self.received_block_rounds[block.author()] {
                (curr_min.min(block.round()), curr_max.max(block.round()))
            } else {
                (block.round(), block.round())
            };
        self.received_block_rounds[block.author()] = Some((min_round, max_round));
    }

    /// Checks if block manager is empty.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.suspended_blocks.is_empty()
            && self.missing_ancestors.is_empty()
            && self.missing_blocks.is_empty()
    }

    /// Returns all the suspended blocks whose causal history we miss hence we can't accept them yet.
    #[cfg(test)]
    fn suspended_blocks(&self) -> Vec<BlockRef> {
        self.suspended_blocks.keys().cloned().collect()
    }
}

// Result of trying to accept one block.
enum TryAcceptResult {
    // The block is accepted. Wraps the block itself.
    Accepted(VerifiedBlock),
    // The block is suspended. Wraps ancestors to be fetched.
    Suspended(BTreeSet<BlockRef>),
    // The block has been processed before and already exists in BlockManager (and is suspended) or
    // in DagState (so has been already accepted). No further processing has been done at this point.
    Processed,
    // When a received block is <= gc_round, then we simply skip its processing as there is no meaning
    // do any action on it or even store it.
    Skipped,
}
