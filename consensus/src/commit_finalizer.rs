// Portions of this file are derived from Mysticeti consensus (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/consensus/core/src/commit_finalizer.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
};

use parking_lot::RwLock;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::task::JoinSet;
use types::committee::Stake;
use types::consensus::{
    block::{BlockAPI, BlockRef, Round, TransactionIndex, VerifiedBlock},
    commit::{CommitIndex, CommittedSubDag, DEFAULT_WAVE_LENGTH},
    context::Context,
    stake_aggregator::{QuorumThreshold, StakeAggregator},
};
use types::error::{ConsensusError, ConsensusResult};

use crate::{dag_state::DagState, transaction_certifier::TransactionCertifier};

/// For transaction T committed at leader round R, when a new leader at round >= R + INDIRECT_REJECT_DEPTH
/// commits and T is still not finalized, T is rejected.
/// NOTE: 3 round is the minimum depth possible for indirect finalization and rejection.
pub(crate) const INDIRECT_REJECT_DEPTH: Round = 3;

/// Handle to CommitFinalizer, for sending CommittedSubDag.
pub(crate) struct CommitFinalizerHandle {
    sender: UnboundedSender<CommittedSubDag>,
}

impl CommitFinalizerHandle {
    // Sends a CommittedSubDag to CommitFinalizer, which will finalize it before sending it to execution.
    pub(crate) fn send(&self, commit: CommittedSubDag) -> ConsensusResult<()> {
        self.sender.send(commit).map_err(|e| {
            tracing::warn!("Failed to send to commit finalizer, probably due to shutdown: {e:?}");
            ConsensusError::Shutdown
        })
    }
}

/// CommitFinalizer accepts a continuous stream of CommittedSubDag and outputs
/// them when they are finalized.
/// In finalized commits, every transaction is either finalized or rejected.
/// It runs in a separate thread, to reduce the load on the core thread.
///
/// Life of a finalized commit:
///
/// For efficiency, finalization happens first for transactions without reject votes (common case).
/// The pending undecided transactions with reject votes are individually finalized or rejected.
/// When there is no more pending transactions, the commit is finalized.
///
/// This is correct because regardless if a commit leader was directly or indirectly committed,
/// every committed block can be considered finalized, because at least one leader certificate of the commit
/// will be committed, which can also serve as a certificate for the block and its transactions.
///
/// From the earliest buffered commit, pending blocks are checked to see if they are now finalized.
/// New finalized blocks are removed from the pending blocks, and its transactions are moved to the
/// finalized, rejected or pending state. If the commit now has no pending blocks or transactions,
/// the commit is finalized and popped from the buffer. The next earliest commit is then processed
/// similarly, until either the buffer becomes empty or a commit with pending blocks or transactions
/// is encountered.
pub struct CommitFinalizer {
    context: Arc<Context>,
    dag_state: Arc<RwLock<DagState>>,
    transaction_certifier: TransactionCertifier,
    commit_sender: UnboundedSender<CommittedSubDag>,

    // Last commit index processed by CommitFinalizer.
    last_processed_commit: Option<CommitIndex>,
    // Commits pending finalization.
    pending_commits: VecDeque<CommitState>,
    // Blocks in the pending commits.
    blocks: Arc<RwLock<BTreeMap<BlockRef, RwLock<BlockState>>>>,
}

impl CommitFinalizer {
    pub fn new(
        context: Arc<Context>,
        dag_state: Arc<RwLock<DagState>>,
        transaction_certifier: TransactionCertifier,
        commit_sender: UnboundedSender<CommittedSubDag>,
    ) -> Self {
        Self {
            context,
            dag_state,
            transaction_certifier,
            commit_sender,
            last_processed_commit: None,
            pending_commits: VecDeque::new(),
            blocks: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    pub(crate) fn start(
        context: Arc<Context>,
        dag_state: Arc<RwLock<DagState>>,
        transaction_certifier: TransactionCertifier,
        commit_sender: UnboundedSender<CommittedSubDag>,
    ) -> CommitFinalizerHandle {
        let processor = Self::new(context, dag_state, transaction_certifier, commit_sender);
        let (sender, receiver) = unbounded_channel();
        let _handle = tokio::spawn(processor.run(receiver));
        CommitFinalizerHandle { sender }
    }

    async fn run(mut self, mut receiver: UnboundedReceiver<CommittedSubDag>) {
        while let Some(committed_sub_dag) = receiver.recv().await {
            let already_finalized = committed_sub_dag.recovered_rejected_transactions;
            let finalized_commits = if !already_finalized {
                self.process_commit(committed_sub_dag).await
            } else {
                vec![committed_sub_dag]
            };
            if !finalized_commits.is_empty() {
                // Transaction certifier state should be GC'ed as soon as new commits are finalized.
                // But this is done outside of process_commit(), because during recovery process_commit()
                // is not called to finalize commits, but GC still needs to run.
                self.try_update_gc_round(finalized_commits.last().unwrap().leader.round);
                let mut dag_state = self.dag_state.write();
                if !already_finalized {
                    // Records rejected transactions in newly finalized commits.
                    for commit in &finalized_commits {
                        dag_state.add_finalized_commit(
                            commit.commit_ref,
                            commit.rejected_transactions_by_block.clone(),
                        );
                    }
                }
                // Commits and committed blocks must be persisted to storage before sending them to Sui
                // to execute their finalized transactions.
                // Commit metadata and uncommitted blocks can be persisted more lazily because they are recoverable.
                // But for simplicity, all unpersisted commits and blocks are flushed to storage.
                dag_state.flush();
            }
            for commit in finalized_commits {
                if let Err(e) = self.commit_sender.send(commit) {
                    tracing::warn!(
                        "Failed to send to commit handler, probably due to shutdown: {e:?}"
                    );
                    return;
                }
            }
        }
    }

    pub async fn process_commit(
        &mut self,
        committed_sub_dag: CommittedSubDag,
    ) -> Vec<CommittedSubDag> {
        if let Some(last_processed_commit) = self.last_processed_commit {
            assert_eq!(last_processed_commit + 1, committed_sub_dag.commit_ref.index);
        }
        self.last_processed_commit = Some(committed_sub_dag.commit_ref.index);

        self.pending_commits.push_back(CommitState::new(committed_sub_dag));

        let mut finalized_commits = vec![];

        // The prerequisite for running direct finalization on a commit is that the commit must
        // have either a quorum of leader certificates in the local DAG, or a committed leader certificate.
        //
        // A leader certificate is a finalization certificate for every block in the commit.
        // When the prerequisite holds, all blocks in the current commit can be considered finalized.
        // And any transaction in the current commit that has not observed reject votes will never be rejected.
        // So these transactions are directly finalized.
        //
        // When a commit is direct, there are a quorum of its leader certificates in the local DAG.
        //
        // When a commit is indirect, it implies one of its leader certificates is in the committed blocks.
        // So a leader certificate must exist in the local DAG as well.
        //
        // When a commit is received through commit sync and processed as certified commit, the commit might
        // not have a leader certificate in the local DAG. So a committed transaction might not observe any reject
        // vote from local DAG, although it will eventually get rejected. To finalize blocks in this commit,
        // there must be another commit with leader round >= 3 (WAVE_LENGTH) rounds above the commit leader.
        // From the indirect commit rule, a leader certificate must exist in committed blocks for the earliest commit.
        for i in 0..self.pending_commits.len() {
            let commit_state = &self.pending_commits[i];
            if commit_state.pending_blocks.is_empty() {
                // The commit has already been processed through direct finalization.
                continue;
            }
            // Direct finalization cannot happen when
            // -  This commit is remote.
            // -  And the latest commit is less than 3 (WAVE_LENGTH) rounds above this commit.
            // In this case, this commit's leader certificate is not guaranteed to be in local DAG.
            if !commit_state.commit.decided_with_local_blocks {
                let last_commit_state = self.pending_commits.back().unwrap();
                if commit_state.commit.leader.round + DEFAULT_WAVE_LENGTH
                    > last_commit_state.commit.leader.round
                {
                    break;
                }
            }
            self.try_direct_finalize_commit(i);
        }
        let direct_finalized_commits = self.pop_finalized_commits();

        finalized_commits.extend(direct_finalized_commits);

        // Indirect finalization: one or more commits cannot be directly finalized.
        // So the pending transactions need to be checked for indirect finalization.
        if !self.pending_commits.is_empty() {
            // Initialize the state of the last added commit for computing indirect finalization.
            //
            // As long as there are remaining commits, even if the last commit has been directly finalized,
            // its state still needs to be initialized here to help indirectly finalize previous commits.
            // This is because the last commit may have been directly finalized, but its previous commits
            // may not have been directly finalized.
            self.link_blocks_in_last_commit();
            self.append_origin_descendants_from_last_commit();
            // Try to indirectly finalize a prefix of the buffered commits.
            // If only one commit remains, it cannot be indirectly finalized because there is no commit afterwards,
            // so it is excluded.
            while self.pending_commits.len() > 1 {
                // Stop indirect finalization when the earliest commit has not been processed
                // through direct finalization.
                if !self.pending_commits[0].pending_blocks.is_empty() {
                    break;
                }
                // Otherwise, try to indirectly finalize the earliest commit.
                self.try_indirect_finalize_first_commit().await;
                let indirect_finalized_commits = self.pop_finalized_commits();
                if indirect_finalized_commits.is_empty() {
                    // No additional commits can be indirectly finalized.
                    break;
                }

                finalized_commits.extend(indirect_finalized_commits);
            }
        }

        finalized_commits
    }

    // Tries directly finalizing transactions in the commit.
    fn try_direct_finalize_commit(&mut self, index: usize) {
        let num_commits = self.pending_commits.len();
        let commit_state = self
            .pending_commits
            .get_mut(index)
            .unwrap_or_else(|| panic!("Commit {} does not exist. len = {}", index, num_commits,));
        // Direct commit means every transaction in the commit can be considered to have a quorum of post-commit certificates,
        // unless the transaction has reject votes that do not reach quorum either.
        assert!(!commit_state.pending_blocks.is_empty());

        let pending_blocks = std::mem::take(&mut commit_state.pending_blocks);
        for (block_ref, num_transactions) in pending_blocks {
            let reject_votes = self.transaction_certifier.get_reject_votes(&block_ref)
                .unwrap_or_else(|| panic!("No vote info found for {block_ref}. It is either incorrectly gc'ed or failed to be recovered after crash."));

            // If a transaction_index does not exist in reject_votes, the transaction has no reject votes.
            // So it is finalized and does not need to be added to pending_transactions.
            for (transaction_index, stake) in reject_votes {
                // If the transaction has > 0 but < 2f+1 reject votes, it is still pending.
                // Otherwise, it is rejected.
                let entry = if stake < self.context.committee.quorum_threshold() {
                    commit_state.pending_transactions.entry(block_ref).or_default()
                } else {
                    commit_state.rejected_transactions.entry(block_ref).or_default()
                };
                entry.insert(transaction_index);
            }
        }
    }

    // Creates an entry in the blocks map for each block in the commit,
    // and have its ancestors link to the block.
    fn link_blocks_in_last_commit(&mut self) {
        let commit_state =
            self.pending_commits.back_mut().unwrap_or_else(|| panic!("No pending commit."));

        // Link blocks in ascending order of round, to ensure ancestor block states are created
        // before they are linked from.
        let mut blocks = commit_state.commit.blocks.clone();
        blocks.sort_by_key(|b| b.round());

        let mut blocks_map = self.blocks.write();
        for block in blocks {
            let block_ref = block.reference();
            // Link ancestors to the block.
            for ancestor in block.ancestors() {
                // Ancestor may not exist in the blocks map if it has been finalized or gc'ed.
                // So skip linking if the ancestor does not exist.
                if let Some(ancestor_block) = blocks_map.get(ancestor) {
                    ancestor_block.write().children.insert(block_ref);
                }
            }
            // Initialize the block state.
            blocks_map.entry(block_ref).or_insert_with(|| RwLock::new(BlockState::new(block)));
        }
    }

    /// To save bandwidth, blocks do not include explicit accept votes on transactions.
    /// Reject votes are included only the first time the block containing the voted-on
    /// transaction is linked in a block. Other first time linked transactions, when
    /// not rejected, are assumed to be accepted. This vote compression rule must also be
    /// applied during vote aggregation.
    ///
    /// Transactions in a block can only be voted on by its immediate descendants.
    /// A block is an **immediate descendant** if it can only link directly to the voted-on
    /// block, without any intermediate blocks from its own authority. Votes from
    /// non-immediate descendants are ignored.
    ///
    /// This rule implies the following optimization is possible: after collecting votes from a block,
    /// we can skip collecting votes from its **origin descendants** (descendant blocks from the
    /// same authority), because their votes would be ignored anyway.
    ///
    /// This function updates the set of origin descendants for all pending blocks using blocks
    /// from the last commit.
    fn append_origin_descendants_from_last_commit(&mut self) {
        let commit_state =
            self.pending_commits.back_mut().unwrap_or_else(|| panic!("No pending commit."));
        let mut committed_blocks = commit_state.commit.blocks.clone();
        committed_blocks.sort_by_key(|b| b.round());
        let blocks_map = self.blocks.read();
        for committed_block in committed_blocks {
            let committed_block_ref = committed_block.reference();
            // Each block must have at least one ancestor.
            // Block verification ensures the first ancestor is from the block's own authority.
            // Also, block verification ensures each authority appears at most once among ancestors.
            let mut origin_ancestor_ref = *blocks_map
                .get(&committed_block_ref)
                .unwrap()
                .read()
                .block
                .ancestors()
                .first()
                .unwrap();
            while origin_ancestor_ref.author == committed_block_ref.author {
                let Some(origin_ancestor_block) = blocks_map.get(&origin_ancestor_ref) else {
                    break;
                };
                origin_ancestor_block.write().origin_descendants.push(committed_block_ref);
                origin_ancestor_ref =
                    *origin_ancestor_block.read().block.ancestors().first().unwrap();
            }
        }
    }

    // Tries indirectly finalizing the buffered commits at the given index.
    async fn try_indirect_finalize_first_commit(&mut self) {
        // Ensure direct finalization has been attempted for the commit.
        assert!(!self.pending_commits.is_empty());
        assert!(self.pending_commits[0].pending_blocks.is_empty());

        // Optional optimization: re-check pending transactions to see if they are rejected by a quorum now.
        self.check_pending_transactions_in_first_commit();

        // Check if remaining pending transactions can be finalized.
        self.try_indirect_finalize_pending_transactions_in_first_commit().await;

        // Check if remaining pending transactions can be indirectly rejected.
        self.try_indirect_reject_pending_transactions_in_first_commit();
    }

    fn check_pending_transactions_in_first_commit(&mut self) {
        let mut all_rejected_transactions: Vec<(BlockRef, Vec<TransactionIndex>)> = vec![];

        // Collect all rejected transactions without modifying state
        for (block_ref, pending_transactions) in &self.pending_commits[0].pending_transactions {
            let reject_votes: BTreeMap<TransactionIndex, Stake> = self
                .transaction_certifier
                .get_reject_votes(block_ref)
                .unwrap_or_else(|| panic!("No vote info found for {block_ref}. It is incorrectly gc'ed or failed to be recovered after crash."))
                .into_iter()
                .collect();
            let mut rejected_transactions = vec![];
            for &transaction_index in pending_transactions {
                // Pending transactions should always have reject votes.
                let reject_stake = reject_votes.get(&transaction_index).copied().unwrap();
                if reject_stake < self.context.committee.quorum_threshold() {
                    // The transaction cannot be rejected yet.
                    continue;
                }
                // Otherwise, mark the transaction for rejection.
                rejected_transactions.push(transaction_index);
            }
            if !rejected_transactions.is_empty() {
                all_rejected_transactions.push((*block_ref, rejected_transactions));
            }
        }

        // Move rejected transactions from pending_transactions.
        for (block_ref, rejected_transactions) in all_rejected_transactions {
            let curr_commit_state = &mut self.pending_commits[0];
            curr_commit_state.remove_pending_transactions(&block_ref, &rejected_transactions);
            curr_commit_state
                .rejected_transactions
                .entry(block_ref)
                .or_default()
                .extend(rejected_transactions);
        }
    }

    async fn try_indirect_finalize_pending_transactions_in_first_commit(&mut self) {
        let pending_blocks: Vec<(BlockRef, BTreeSet<TransactionIndex>)> = self.pending_commits[0]
            .pending_transactions
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        // Number of blocks to process in each task.
        const BLOCKS_PER_INDIRECT_COMMIT_TASK: usize = 8;

        // Process chunks in parallel.
        let mut all_finalized_transactions = vec![];
        let mut join_set = JoinSet::new();
        // TODO(fastpath): investigate using a cost based batching,
        // for example each block has cost num authorities + pending_transactions.len().
        for chunk in pending_blocks.chunks(BLOCKS_PER_INDIRECT_COMMIT_TASK) {
            let context = self.context.clone();
            let blocks = self.blocks.clone();
            let chunk: Vec<(BlockRef, BTreeSet<TransactionIndex>)> = chunk.to_vec();

            join_set.spawn(tokio::task::spawn_blocking(move || {
                let mut chunk_results = Vec::new();

                for (block_ref, pending_transactions) in chunk {
                    let finalized = Self::try_indirect_finalize_pending_transactions_in_block(
                        &context,
                        &blocks,
                        block_ref,
                        pending_transactions,
                    );

                    if !finalized.is_empty() {
                        chunk_results.push((block_ref, finalized));
                    }
                }

                chunk_results
            }));
        }

        // Collect results from all chunks
        while let Some(result) = join_set.join_next().await {
            let e = match result {
                Ok(blocking_result) => match blocking_result {
                    Ok(chunk_results) => {
                        all_finalized_transactions.extend(chunk_results);
                        continue;
                    }
                    Err(e) => e,
                },
                Err(e) => e,
            };
            if e.is_panic() {
                std::panic::resume_unwind(e.into_panic());
            }
            tracing::info!("Process likely shutting down: {:?}", e);
            // Ok to return. No potential inconsistency in state.
            return;
        }

        for (block_ref, finalized_transactions) in all_finalized_transactions {
            // Remove finalized transactions from pending transactions.
            self.pending_commits[0]
                .remove_pending_transactions(&block_ref, &finalized_transactions);
        }
    }

    fn try_indirect_reject_pending_transactions_in_first_commit(&mut self) {
        let curr_leader_round = self.pending_commits[0].commit.leader.round;
        let last_commit_leader_round = self.pending_commits.back().unwrap().commit.leader.round;
        if curr_leader_round + INDIRECT_REJECT_DEPTH <= last_commit_leader_round {
            let curr_commit_state = &mut self.pending_commits[0];
            // This function is called after trying to indirectly finalize pending blocks.
            // When last commit leader round is INDIRECT_REJECT_DEPTH rounds higher or more,
            // all pending blocks should have been finalized.
            assert!(curr_commit_state.pending_blocks.is_empty());
            // This function is called after trying to indirectly finalize pending transactions.
            // All remaining pending transactions, since they are not finalized, should now be
            // indirectly rejected.
            let pending_transactions = std::mem::take(&mut curr_commit_state.pending_transactions);
            for (block_ref, pending_transactions) in pending_transactions {
                curr_commit_state
                    .rejected_transactions
                    .entry(block_ref)
                    .or_default()
                    .extend(pending_transactions);
            }
        }
    }

    // Returns the indices of the requested pending transactions that are indirectly finalized.
    // This function is used for checking finalization of transactions, so it must traverse
    // all blocks which can contribute to the requested transactions' finalizations.
    fn try_indirect_finalize_pending_transactions_in_block(
        context: &Arc<Context>,
        blocks: &Arc<RwLock<BTreeMap<BlockRef, RwLock<BlockState>>>>,
        pending_block_ref: BlockRef,
        pending_transactions: BTreeSet<TransactionIndex>,
    ) -> Vec<TransactionIndex> {
        if pending_transactions.is_empty() {
            return vec![];
        }
        let mut accept_votes: BTreeMap<TransactionIndex, StakeAggregator<QuorumThreshold>> =
            pending_transactions
                .into_iter()
                .map(|transaction_index| (transaction_index, StakeAggregator::new()))
                .collect();
        let mut finalized_transactions = vec![];
        let blocks_map = blocks.read();
        // Use BTreeSet to ensure always visit blocks in the earliest round.
        let mut to_visit_blocks =
            blocks_map.get(&pending_block_ref).unwrap().read().children.clone();
        // Blocks that have been visited.
        let mut visited = BTreeSet::new();
        // Blocks where votes and origin descendants should be ignored for processing.
        let mut ignored = BTreeSet::new();
        // Traverse children blocks breadth-first and accumulate accept votes for pending transactions.
        while let Some(curr_block_ref) = to_visit_blocks.pop_first() {
            if !visited.insert(curr_block_ref) {
                continue;
            }
            let curr_block_state = blocks_map.get(&curr_block_ref).unwrap_or_else(|| panic!("Block {curr_block_ref} is either incorrectly gc'ed or failed to be recovered after crash.")).read();
            // Ignore info from the block if its direct ancestor has been processed.
            if ignored.insert(curr_block_ref) {
                // Skip collecting votes from origin descendants of current block.
                // Votes from origin descendants of current block do not count for this transactions.
                // Consider this case: block B is an origin descendant of block A (from the same authority),
                // and both blocks A and B link to another block C.
                // Only B's implicit and explicit transaction votes on C are considered.
                // None of A's implicit or explicit transaction votes on C should be considered.
                ignored.extend(curr_block_state.origin_descendants.iter());
                // Get reject votes from current block to the pending block.
                let curr_block_reject_votes = curr_block_state
                    .reject_votes
                    .get(&pending_block_ref)
                    .cloned()
                    .unwrap_or_default();
                // Because of lifetime, first collect finalized transactions, and then remove them from accept_votes.
                let mut newly_finalized = vec![];
                for (index, stake) in &mut accept_votes {
                    // Skip if the transaction has been rejected by the current block.
                    if curr_block_reject_votes.contains(index) {
                        continue;
                    }
                    // Skip if the total stake has not reached quorum.
                    if !stake.add(curr_block_ref.author, &context.committee) {
                        continue;
                    }
                    newly_finalized.push(*index);
                    finalized_transactions.push(*index);
                }
                // There is no need to aggregate additional votes for already finalized transactions.
                for index in newly_finalized {
                    accept_votes.remove(&index);
                }
                // End traversing if all blocks and requested transactions have reached quorum.
                if accept_votes.is_empty() {
                    break;
                }
            }
            // Add additional children blocks to visit.
            to_visit_blocks
                .extend(curr_block_state.children.iter().filter(|b| !visited.contains(*b)));
        }
        finalized_transactions
    }

    fn pop_finalized_commits(&mut self) -> Vec<CommittedSubDag> {
        let mut finalized_commits = vec![];

        while let Some(commit_state) = self.pending_commits.front() {
            if !commit_state.pending_blocks.is_empty()
                || !commit_state.pending_transactions.is_empty()
            {
                // The commit is not finalized yet.
                break;
            }

            // Pop the finalized commit and set its rejected transactions.
            let commit_state = self.pending_commits.pop_front().unwrap();
            let mut commit = commit_state.commit;
            for (block_ref, rejected_transactions) in commit_state.rejected_transactions {
                commit
                    .rejected_transactions_by_block
                    .insert(block_ref, rejected_transactions.into_iter().collect());
            }

            // Clean up committed blocks.
            let mut blocks_map = self.blocks.write();
            for block in commit.blocks.iter() {
                blocks_map.remove(&block.reference());
            }

            let round_delay = if let Some(last_commit_state) = self.pending_commits.back() {
                last_commit_state.commit.leader.round - commit.leader.round
            } else {
                0
            };

            finalized_commits.push(commit);
        }

        finalized_commits
    }

    fn try_update_gc_round(&mut self, last_finalized_commit_round: Round) {
        // GC TransactionCertifier state only with finalized commits, to ensure unfinalized transactions
        // can access their reject votes from TransactionCertifier.
        let gc_round = self.dag_state.read().calculate_gc_round(last_finalized_commit_round);
        self.transaction_certifier.run_gc(gc_round);
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.pending_commits.is_empty() && self.blocks.read().is_empty()
    }
}

struct CommitState {
    commit: CommittedSubDag,
    // Blocks pending finalization, mapped to the number of transactions in the block.
    // This field is populated by all blocks in the commit, before direct finalization.
    // After direct finalization, this field becomes empty.
    pending_blocks: BTreeMap<BlockRef, usize>,
    // Transactions pending indirect finalization.
    // This field is populated after direct finalization, if pending transactions exist.
    // Values in this field are removed as transactions are indirectly finalized or directly rejected.
    // When both pending_blocks and pending_transactions are empty, the commit is finalized.
    pending_transactions: BTreeMap<BlockRef, BTreeSet<TransactionIndex>>,
    // Transactions rejected by a quorum or indirectly, per block.
    rejected_transactions: BTreeMap<BlockRef, BTreeSet<TransactionIndex>>,
}

impl CommitState {
    fn new(commit: CommittedSubDag) -> Self {
        let pending_blocks: BTreeMap<_, _> =
            commit.blocks.iter().map(|b| (b.reference(), b.transactions().len())).collect();
        assert!(!pending_blocks.is_empty());
        Self {
            commit,
            pending_blocks,
            pending_transactions: BTreeMap::new(),
            rejected_transactions: BTreeMap::new(),
        }
    }

    fn remove_pending_transactions(
        &mut self,
        block_ref: &BlockRef,
        transactions: &[TransactionIndex],
    ) {
        let Some(block_pending_txns) = self.pending_transactions.get_mut(block_ref) else {
            return;
        };
        for t in transactions {
            block_pending_txns.remove(t);
        }
        if block_pending_txns.is_empty() {
            self.pending_transactions.remove(block_ref);
        }
    }
}

struct BlockState {
    // Content of the block.
    block: VerifiedBlock,
    // Blocks which has an explicit ancestor linking to this block.
    children: BTreeSet<BlockRef>,
    // Reject votes casted by this block, and by linked ancestors from the same authority.
    reject_votes: BTreeMap<BlockRef, BTreeSet<TransactionIndex>>,
    // Other committed blocks that are origin descendants of this block.
    origin_descendants: Vec<BlockRef>,
}

impl BlockState {
    fn new(block: VerifiedBlock) -> Self {
        let reject_votes: BTreeMap<_, _> = block
            .transaction_votes()
            .iter()
            .map(|v| (v.block_ref, v.rejects.clone().into_iter().collect()))
            .collect();
        // With at most 4 pending commits and assume 2 origin descendants per commit,
        // there will be at most 8 origin descendants.
        let origin_descendants = Vec::with_capacity(8);
        Self { block, children: BTreeSet::new(), reject_votes, origin_descendants }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::RwLock;
    use tokio::sync::mpsc::unbounded_channel;
    use types::consensus::{
        block::{BlockAPI, TransactionIndex},
        context::Context,
    };
    use types::storage::consensus::mem_store::MemStore;

    use crate::{
        block_verifier::NoopBlockVerifier, dag_state::DagState, test_dag_builder::DagBuilder,
        transaction_certifier::TransactionCertifier,
    };

    use super::CommitFinalizer;

    /// Build a fully connected DAG, commit through the pipeline, and verify all
    /// blocks are directly finalized with no rejected transactions.
    #[tokio::test]
    async fn test_direct_finalize_no_reject_votes() {
        let num_authorities = 4;
        let (context, _keys) = Context::new_for_test(num_authorities);
        let context = Arc::new(context);

        let mem_store = Arc::new(MemStore::new());
        let dag_state = Arc::new(RwLock::new(DagState::new(context.clone(), mem_store.clone())));

        let (blocks_sender, _blocks_receiver) = unbounded_channel();
        let transaction_certifier = TransactionCertifier::new(
            context.clone(),
            Arc::new(NoopBlockVerifier {}),
            dag_state.clone(),
            blocks_sender,
        );

        let (commit_sender, _commit_receiver) = unbounded_channel();
        let mut commit_finalizer = CommitFinalizer::new(
            context.clone(),
            dag_state.clone(),
            transaction_certifier.clone(),
            commit_sender,
        );

        // Build a fully connected DAG with transactions, but no reject votes.
        let num_rounds: u32 = 10;
        let mut dag_builder = DagBuilder::new(context.clone());
        dag_builder
            .layers(1..=num_rounds)
            .num_transactions(4)
            .build()
            .persist_layers(dag_state.clone());

        // Feed all blocks to the transaction certifier with no own-reject-votes.
        // This ensures get_reject_votes() will return Some for each block.
        transaction_certifier.add_voted_blocks(
            dag_builder.all_blocks().iter().map(|b| (b.clone(), vec![])).collect(),
        );

        // Get committed sub-dags from the DAG builder.
        let committed = dag_builder.get_sub_dag_and_commits(1..=num_rounds);
        assert!(!committed.is_empty(), "Expected at least one committed sub-dag");

        // Process each committed sub-dag through the finalizer.
        let mut all_finalized = vec![];
        for (sub_dag, _commit) in committed {
            let finalized = commit_finalizer.process_commit(sub_dag).await;
            all_finalized.extend(finalized);
        }

        // Verify that commits were finalized.
        assert!(!all_finalized.is_empty(), "Expected at least one finalized commit");

        // Verify that no transactions were rejected in any finalized commit.
        for commit in &all_finalized {
            assert!(
                commit.rejected_transactions_by_block.is_empty(),
                "Expected no rejected transactions in commit at leader {:?}, but found: {:?}",
                commit.leader,
                commit.rejected_transactions_by_block,
            );
        }

        // Verify that each finalized commit has blocks with transactions.
        for commit in &all_finalized {
            assert!(
                !commit.blocks.is_empty(),
                "Expected blocks in finalized commit at leader {:?}",
                commit.leader,
            );
            for block in &commit.blocks {
                // Each block from round > 0 should have 4 transactions.
                if block.round() > 0 {
                    assert_eq!(
                        block.transactions().len(),
                        4,
                        "Expected 4 transactions in block {:?}",
                        block.reference(),
                    );
                }
            }
        }

        // After processing all commits, the finalizer should be empty
        // (no pending commits remaining).
        assert!(
            commit_finalizer.is_empty(),
            "Expected commit finalizer to be empty after processing all commits"
        );
    }

    /// Build blocks with reject votes and verify that transactions with a quorum
    /// of reject votes are marked as rejected after finalization.
    #[tokio::test]
    async fn test_direct_finalize_with_reject_votes() {
        let num_authorities = 4;
        let (context, _keys) = Context::new_for_test(num_authorities);
        let context = Arc::new(context);

        let mem_store = Arc::new(MemStore::new());
        let dag_state = Arc::new(RwLock::new(DagState::new(context.clone(), mem_store.clone())));

        let (blocks_sender, _blocks_receiver) = unbounded_channel();
        let transaction_certifier = TransactionCertifier::new(
            context.clone(),
            Arc::new(NoopBlockVerifier {}),
            dag_state.clone(),
            blocks_sender,
        );

        let (commit_sender, _commit_receiver) = unbounded_channel();
        let mut commit_finalizer = CommitFinalizer::new(
            context.clone(),
            dag_state.clone(),
            transaction_certifier.clone(),
            commit_sender,
        );

        // Build a DAG with transactions AND reject votes.
        // Using 100% rejection rate means every block votes to reject all
        // transactions of its ancestors. With 4 authorities, each block in
        // round R gets reject votes from all 4 blocks in round R+1, which
        // exceeds the 2f+1=3 quorum threshold.
        let num_rounds: u32 = 10;
        let mut dag_builder = DagBuilder::new(context.clone());
        dag_builder
            .layers(1..=num_rounds)
            .num_transactions(4)
            .rejected_transactions_pct(100, Some(42))
            .build()
            .persist_layers(dag_state.clone());

        // Feed all blocks with no own-reject-votes (the reject votes are encoded
        // in the blocks' transaction_votes fields via the DagBuilder).
        transaction_certifier.add_voted_blocks(
            dag_builder.all_blocks().iter().map(|b| (b.clone(), vec![])).collect(),
        );

        // Get committed sub-dags.
        let committed = dag_builder.get_sub_dag_and_commits(1..=num_rounds);
        assert!(!committed.is_empty(), "Expected at least one committed sub-dag");

        // Process each committed sub-dag through the finalizer.
        let mut all_finalized = vec![];
        for (sub_dag, _commit) in committed {
            let finalized = commit_finalizer.process_commit(sub_dag).await;
            all_finalized.extend(finalized);
        }

        assert!(!all_finalized.is_empty(), "Expected at least one finalized commit");

        // Verify that some transactions were rejected. With 100% rejection rate
        // and 4 authorities (quorum = 3 stake), blocks that have a subsequent
        // round of blocks voting on them should have all transactions rejected.
        // However, blocks in the last committed round may not have subsequent
        // rounds voting on them yet, so they might not have reached quorum.
        let mut total_rejected_transactions = 0usize;
        for commit in &all_finalized {
            for rejected_indices in commit.rejected_transactions_by_block.values() {
                total_rejected_transactions += rejected_indices.len();
            }
        }

        // With 100% reject votes and enough rounds, we expect at least some
        // transactions to be rejected (those in early rounds that have a full
        // subsequent round of reject votes).
        assert!(
            total_rejected_transactions > 0,
            "Expected some rejected transactions with 100% reject vote rate, but found none"
        );

        // Verify that rejected transaction indices are valid (within bounds).
        for commit in &all_finalized {
            for (block_ref, rejected_indices) in &commit.rejected_transactions_by_block {
                // Find the block in the commit's blocks.
                let block = commit.blocks.iter().find(|b| b.reference() == *block_ref);
                if let Some(block) = block {
                    let num_txns = block.transactions().len() as TransactionIndex;
                    for &idx in rejected_indices {
                        assert!(
                            idx < num_txns,
                            "Rejected transaction index {} is out of bounds for block {:?} with {} transactions",
                            idx,
                            block_ref,
                            num_txns,
                        );
                    }
                }
            }
        }
    }
}
