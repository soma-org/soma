use std::sync::Arc;

use crate::dag_state::DagState;
use itertools::Itertools;
use parking_lot::RwLock;
use types::committee::Stake;
use types::consensus::{
    block::{BlockAPI, BlockRef, BlockTimestampMs, Round, VerifiedBlock},
    commit::{Commit, CommittedSubDag, TrustedCommit, sort_sub_dag_blocks},
    context::Context,
};

/// The `StorageAPI` trait provides an interface for the block store and has been
/// mostly introduced for allowing to inject the test store in `DagBuilder`.
pub(crate) trait BlockStoreAPI {
    fn get_blocks(&self, refs: &[BlockRef]) -> Vec<Option<VerifiedBlock>>;

    fn gc_round(&self) -> Round;

    fn set_committed(&mut self, block_ref: &BlockRef) -> bool;

    fn is_committed(&self, block_ref: &BlockRef) -> bool;
}

impl BlockStoreAPI
    for parking_lot::lock_api::RwLockWriteGuard<'_, parking_lot::RawRwLock, DagState>
{
    fn get_blocks(&self, refs: &[BlockRef]) -> Vec<Option<VerifiedBlock>> {
        DagState::get_blocks(self, refs)
    }

    fn gc_round(&self) -> Round {
        DagState::gc_round(self)
    }

    fn set_committed(&mut self, block_ref: &BlockRef) -> bool {
        DagState::set_committed(self, block_ref)
    }

    fn is_committed(&self, block_ref: &BlockRef) -> bool {
        DagState::is_committed(self, block_ref)
    }
}

/// Expand a committed sequence of leader into a sequence of sub-dags.
#[derive(Clone)]
pub struct Linearizer {
    /// In memory block store representing the dag state
    context: Arc<Context>,
    dag_state: Arc<RwLock<DagState>>,
}

impl Linearizer {
    pub fn new(context: Arc<Context>, dag_state: Arc<RwLock<DagState>>) -> Self {
        Self { context, dag_state }
    }

    /// Collect the sub-dag and the corresponding commit from a specific leader excluding any duplicates or
    /// blocks that have already been committed (within previous sub-dags).
    fn collect_sub_dag_and_commit(
        &mut self,
        leader_block: VerifiedBlock,
    ) -> (CommittedSubDag, TrustedCommit) {
        // Grab latest commit state from dag state
        let mut dag_state = self.dag_state.write();
        let last_commit_index = dag_state.last_commit_index();
        let last_commit_digest = dag_state.last_commit_digest();
        let last_commit_timestamp_ms = dag_state.last_commit_timestamp_ms();

        // Now linearize the sub-dag starting from the leader block
        let to_commit = Self::linearize_sub_dag(leader_block.clone(), &mut dag_state);

        let timestamp_ms = Self::calculate_commit_timestamp(
            &self.context,
            &mut dag_state,
            &leader_block,
            last_commit_timestamp_ms,
        );

        drop(dag_state);

        // Create the Commit.
        let commit = Commit::new(
            last_commit_index + 1,
            last_commit_digest,
            timestamp_ms,
            leader_block.reference(),
            to_commit.iter().map(|block| block.reference()).collect::<Vec<_>>(),
        );
        let serialized =
            commit.serialize().unwrap_or_else(|e| panic!("Failed to serialize commit: {}", e));
        let commit = TrustedCommit::new_trusted(commit, serialized);

        // Create the corresponding committed sub dag
        let sub_dag = CommittedSubDag::new(
            leader_block.reference(),
            to_commit,
            timestamp_ms,
            commit.reference(),
        );

        (sub_dag, commit)
    }

    /// Calculates the commit's timestamp. The timestamp will be calculated as the median of leader's parents (leader.round - 1)
    /// timestamps by stake. To ensure that commit timestamp monotonicity is respected it is compared against the `last_commit_timestamp_ms`
    /// and the maximum of the two is returned.
    pub(crate) fn calculate_commit_timestamp(
        context: &Context,
        dag_state: &mut impl BlockStoreAPI,
        leader_block: &VerifiedBlock,
        last_commit_timestamp_ms: BlockTimestampMs,
    ) -> BlockTimestampMs {
        let timestamp_ms = {
            // Select leaders' parent blocks.
            let block_refs = leader_block
                .ancestors()
                .iter()
                .filter(|block_ref| block_ref.round == leader_block.round() - 1)
                .cloned()
                .collect::<Vec<_>>();
            // Get the blocks from dag state which should not fail.
            let blocks = dag_state
                .get_blocks(&block_refs)
                .into_iter()
                .map(|block_opt| block_opt.expect("We should have all blocks in dag state."));
            median_timestamp_by_stake(context, blocks).unwrap_or_else(|e| {
                panic!(
                    "Cannot compute median timestamp for leader block {:?} ancestors: {}",
                    leader_block, e
                )
            })
        };

        // Always make sure that commit timestamps are monotonic, so override if necessary.
        timestamp_ms.max(last_commit_timestamp_ms)
    }

    pub(crate) fn linearize_sub_dag(
        leader_block: VerifiedBlock,
        dag_state: &mut impl BlockStoreAPI,
    ) -> Vec<VerifiedBlock> {
        // The GC round here is calculated based on the last committed round of the leader block. The algorithm will attempt to
        // commit blocks up to this GC round. Once this commit has been processed and written to DagState, then gc round will update
        // and on the processing of the next commit we'll have it already updated, so no need to do any gc_round recalculations here.
        // We just use whatever is currently in DagState.
        let gc_round: Round = dag_state.gc_round();
        let leader_block_ref = leader_block.reference();
        let mut buffer = vec![leader_block];
        let mut to_commit = Vec::new();

        // Perform the recursion without stopping at the highest round round that has been committed per authority. Instead it will
        // allow to commit blocks that are lower than the highest committed round for an authority but higher than gc_round.
        assert!(
            dag_state.set_committed(&leader_block_ref),
            "Leader block with reference {:?} attempted to be committed twice",
            leader_block_ref
        );

        while let Some(x) = buffer.pop() {
            to_commit.push(x.clone());

            let ancestors: Vec<VerifiedBlock> = dag_state
                .get_blocks(
                    &x.ancestors()
                        .iter()
                        .copied()
                        .filter(|ancestor| {
                            ancestor.round > gc_round && !dag_state.is_committed(ancestor)
                        })
                        .collect::<Vec<_>>(),
                )
                .into_iter()
                .map(|ancestor_opt| {
                    ancestor_opt.expect("We should have all uncommitted blocks in dag state.")
                })
                .collect();

            for ancestor in ancestors {
                buffer.push(ancestor.clone());
                assert!(
                    dag_state.set_committed(&ancestor.reference()),
                    "Block with reference {:?} attempted to be committed twice",
                    ancestor.reference()
                );
            }
        }

        // The above code should have not yielded any blocks that are <= gc_round, but just to make sure that we'll never
        // commit anything that should be garbage collected we attempt to prune here as well.
        assert!(
            to_commit.iter().all(|block| block.round() > gc_round),
            "No blocks <= {gc_round} should be committed. Leader round {}, blocks {to_commit:?}.",
            leader_block_ref
        );

        // Sort the blocks of the sub-dag blocks
        sort_sub_dag_blocks(&mut to_commit);

        to_commit
    }

    // This function should be called whenever a new commit is observed. This will
    // iterate over the sequence of committed leaders and produce a list of committed
    // sub-dags.
    pub fn handle_commit(&mut self, committed_leaders: Vec<VerifiedBlock>) -> Vec<CommittedSubDag> {
        if committed_leaders.is_empty() {
            return vec![];
        }

        let mut committed_sub_dags = vec![];
        for leader_block in committed_leaders {
            // Collect the sub-dag generated using each of these leaders and the corresponding commit.
            let (sub_dag, commit) = self.collect_sub_dag_and_commit(leader_block);

            // Buffer commit in dag state for persistence later.
            // This also updates the last committed rounds.
            self.dag_state.write().add_commit(commit.clone());

            committed_sub_dags.push(sub_dag);
        }

        committed_sub_dags
    }
}

/// Computes the median timestamp of the blocks weighted by the stake of their authorities.
/// This function assumes each block comes from a different authority of the same round.
/// Error is returned if no blocks are provided or total stake is less than quorum threshold.
pub(crate) fn median_timestamp_by_stake(
    context: &Context,
    blocks: impl Iterator<Item = VerifiedBlock>,
) -> Result<BlockTimestampMs, String> {
    let mut total_stake = 0;
    let mut timestamps = vec![];
    for block in blocks {
        let stake = context.committee.stake_by_index(block.author());
        timestamps.push((block.timestamp_ms(), stake));
        total_stake += stake;
    }

    if timestamps.is_empty() {
        return Err("No blocks provided".to_string());
    }
    if total_stake < context.committee.quorum_threshold() {
        return Err(format!(
            "Total stake {} < quorum threshold {}",
            total_stake,
            context.committee.quorum_threshold()
        )
        .to_string());
    }

    Ok(median_timestamps_by_stake_inner(timestamps, total_stake))
}

fn median_timestamps_by_stake_inner(
    mut timestamps: Vec<(BlockTimestampMs, Stake)>,
    total_stake: Stake,
) -> BlockTimestampMs {
    timestamps.sort_by_key(|(ts, _)| *ts);

    let mut cumulative_stake = 0;
    for (ts, stake) in &timestamps {
        cumulative_stake += stake;
        if cumulative_stake > total_stake / 2 {
            return *ts;
        }
    }

    timestamps.last().unwrap().0
}
