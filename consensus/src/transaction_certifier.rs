// Portions of this file are derived from Mysticeti consensus (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/consensus/core/src/transaction_certifier.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use crate::{block_verifier::BlockVerifier, dag_state::DagState};
use parking_lot::RwLock;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, info};
use types::committee::Stake;
use types::consensus::{
    block::{
        BlockAPI as _, BlockRef, BlockTransactionVotes, CertifiedBlock, CertifiedBlocksOutput,
        GENESIS_ROUND, Round, TransactionIndex, VerifiedBlock,
    },
    context::Context,
    stake_aggregator::{QuorumThreshold, StakeAggregator},
};

/// TransactionCertifier has the following purposes:
/// 1. Certifies transactions and sends them to execute on the fastpath.
/// 2. Keeps track of own votes on transactions, and allows the votes to be retrieved
///    later in core after acceptance of the blocks containing the transactions.
/// 3. Aggregates reject votes on transactions, and allows the aggregated votes
///    to be retrieved during post-commit finalization.
///
/// A transaction is certified if a quorum of authorities in the causal history of a proposed block
/// vote to accept the transaction. Accept votes are implicit in blocks: if a transaction is in
/// the causal history of a block and the block does not vote to reject it, the block
/// is considered to vote to accept the transaction. Transaction finalization are eventually resolved
/// post commit, by checking if there is a certification of the transaction in the causal history
/// of the leader. So only accept votes are only considered if they are in the causal history of own
/// proposed blocks.
///
/// A transaction is rejected if a quorum of authorities vote to reject it. When this happens, it is
/// guaranteed that no validator can observe a certification of the transaction, with <= f malicious
/// stake.
///
/// A block is certified if every transaction in the block is either certified or rejected.
/// TransactionCertifier outputs certified blocks.
///
/// The invariant between TransactionCertifier and post-commit finalization is that if a quorum of
/// authorities certified a transaction for fastpath and executed it, then the transaction
/// must also be finalized post consensus commit. The reverse is not true though, because
/// fastpath execution is only a latency optimization, and not required for correctness.
#[derive(Clone)]
pub struct TransactionCertifier {
    // The state of blocks being voted on and certified.
    certifier_state: Arc<RwLock<CertifierState>>,
    // Verify transactions during recovery.
    block_verifier: Arc<dyn BlockVerifier>,
    // The state of the DAG.
    dag_state: Arc<RwLock<DagState>>,
    // An unbounded channel to output certified blocks to Sui consensus block handler.
    certified_blocks_sender: UnboundedSender<CertifiedBlocksOutput>,
}

impl TransactionCertifier {
    pub fn new(
        context: Arc<Context>,
        block_verifier: Arc<dyn BlockVerifier>,
        dag_state: Arc<RwLock<DagState>>,
        certified_blocks_sender: UnboundedSender<CertifiedBlocksOutput>,
    ) -> Self {
        Self {
            certifier_state: Arc::new(RwLock::new(CertifierState::new(context))),
            block_verifier,
            dag_state,
            certified_blocks_sender,
        }
    }

    /// Recovers all blocks from DB after the given round.
    ///
    /// This is useful for initializing the certifier state
    /// for future commits and block proposals.
    pub(crate) fn recover_blocks_after_round(&self, after_round: Round) {
        let context = self.certifier_state.read().context.clone();

        let store = self.dag_state.read().store().clone();

        let recovery_start_round = after_round + 1;
        info!("Recovering certifier state from round {}", recovery_start_round,);

        let authorities =
            context.committee.authorities().map(|(index, _)| index).collect::<Vec<_>>();
        for authority_index in authorities {
            let blocks =
                store.scan_blocks_by_author(authority_index, recovery_start_round).unwrap();
            info!(
                "Recovered and voting on {} blocks from authority {}",
                blocks.len(),
                authority_index,
            );
            self.recover_and_vote_on_blocks(blocks);
        }
    }

    /// Recovers and potentially votes on the given blocks.
    ///
    /// Because own votes on blocks are not stored, during recovery it is necessary to vote on
    /// input blocks that are above GC round and have not been included before, which can be
    /// included in a future proposed block.
    ///
    /// In addition, add_voted_blocks() will eventually process reject votes contained in the input blocks.
    pub(crate) fn recover_and_vote_on_blocks(&self, blocks: Vec<VerifiedBlock>) {
        let should_vote_blocks = {
            let dag_state = self.dag_state.read();
            let gc_round = dag_state.gc_round();
            blocks
                .iter()
                // Must make sure the block is above GC round before calling has_been_included().
                .map(|b| b.round() > gc_round && !dag_state.has_been_included(&b.reference()))
                .collect::<Vec<_>>()
        };
        let voted_blocks = blocks
            .into_iter()
            .zip(should_vote_blocks)
            .map(|(b, should_vote)| {
                if !should_vote {
                    // Voting is unnecessary for blocks already included in own proposed blocks,
                    // or outside of local DAG GC bound.
                    (b, vec![])
                } else {
                    // Voting is needed for blocks above GC round and not yet included in own proposed blocks.
                    // A block proposal can include the input block later and retries own votes on it.
                    let reject_transaction_votes =
                        self.block_verifier.vote(&b).unwrap_or_else(|e| {
                            panic!("Failed to vote on block during recovery: {}", e)
                        });
                    (b, reject_transaction_votes)
                }
            })
            .collect::<Vec<_>>();
        self.certifier_state.write().add_voted_blocks(voted_blocks);
        // Do not send certified blocks to the fastpath output channel during recovery,
        // because these transactions could have been executed and fastpath latency optimization is
        // unnecessary for recovered transactions.
    }

    /// Stores own reject votes on input blocks, and aggregates reject votes from the input blocks.
    /// Newly certified blocks are sent to the fastpath output channel.
    pub fn add_voted_blocks(&self, voted_blocks: Vec<(VerifiedBlock, Vec<TransactionIndex>)>) {
        let certified_blocks = self.certifier_state.write().add_voted_blocks(voted_blocks);
        self.send_certified_blocks(certified_blocks);
    }

    /// Aggregates accept votes from the own proposed block.
    /// Newly certified blocks are sent to the fastpath output channel.
    pub(crate) fn add_proposed_block(&self, proposed_block: VerifiedBlock) {
        let certified_blocks = self.certifier_state.write().add_proposed_block(proposed_block);
        self.send_certified_blocks(certified_blocks);
    }

    // Sends certified blocks to the fastpath output channel.
    fn send_certified_blocks(&self, certified_blocks: Vec<CertifiedBlock>) {
        if certified_blocks.is_empty() {
            return;
        }
        if let Err(e) =
            self.certified_blocks_sender.send(CertifiedBlocksOutput { blocks: certified_blocks })
        {
            tracing::warn!("Failed to send certified blocks: {:?}", e);
        }
    }

    /// Retrieves own votes on peer block transactions.
    pub(crate) fn get_own_votes(&self, block_refs: Vec<BlockRef>) -> Vec<BlockTransactionVotes> {
        let mut votes = vec![];
        let certifier_state = self.certifier_state.read();
        for block_ref in block_refs {
            if block_ref.round <= certifier_state.gc_round {
                continue;
            }
            let vote_info = certifier_state.votes.get(&block_ref).unwrap_or_else(|| {
                panic!("Ancestor block {} not found in certifier state", block_ref)
            });
            if !vote_info.own_reject_txn_votes.is_empty() {
                votes.push(BlockTransactionVotes {
                    block_ref,
                    rejects: vote_info.own_reject_txn_votes.clone(),
                });
            }
        }
        votes
    }

    /// Retrieves transactions in the block that have received reject votes, and the total stake of the votes.
    /// TransactionIndex not included in the output has no reject votes.
    /// Returns None if no information is found for the block.
    pub(crate) fn get_reject_votes(
        &self,
        block_ref: &BlockRef,
    ) -> Option<Vec<(TransactionIndex, Stake)>> {
        let accumulated_reject_votes = self
            .certifier_state
            .read()
            .votes
            .get(block_ref)?
            .reject_txn_votes
            .iter()
            .map(|(idx, stake_agg)| (*idx, stake_agg.stake()))
            .collect::<Vec<_>>();
        Some(accumulated_reject_votes)
    }

    /// Runs garbage collection on the internal state by removing data for blocks <= gc_round,
    /// and updates the GC round for the certifier.
    ///
    /// IMPORTANT: the gc_round used here can trail the latest gc_round from DagState.
    /// This is because the gc round here is determined by CommitFinalizer, which needs to process
    /// commits before the latest commit in DagState. Reject votes received by transactions below
    /// local DAG gc_round may still need to be accessed from CommitFinalizer.
    pub(crate) fn run_gc(&self, gc_round: Round) {
        let dag_state_gc_round = self.dag_state.read().gc_round();
        assert!(
            gc_round <= dag_state_gc_round,
            "TransactionCertifier cannot GC higher than DagState GC round ({} > {})",
            gc_round,
            dag_state_gc_round
        );
        self.certifier_state.write().update_gc_round(gc_round);
    }
}

/// CertifierState keeps track of votes received by each transaction and block,
/// and helps determine if votes reach a quorum. Reject votes can start accumulating
/// even before the target block is received by this authority.
struct CertifierState {
    context: Arc<Context>,

    // Maps received blocks' refs to votes on those blocks from other blocks.
    // Even if a block has no reject votes on its transactions, it still has an entry here.
    votes: BTreeMap<BlockRef, VoteInfo>,

    // Highest round where blocks are GC'ed.
    gc_round: Round,
}

impl CertifierState {
    fn new(context: Arc<Context>) -> Self {
        Self { context, votes: BTreeMap::new(), gc_round: GENESIS_ROUND }
    }

    fn add_voted_blocks(
        &mut self,
        voted_blocks: Vec<(VerifiedBlock, Vec<TransactionIndex>)>,
    ) -> Vec<CertifiedBlock> {
        let mut certified_blocks = vec![];
        for (voted_block, reject_txn_votes) in voted_blocks {
            let blocks = self.add_voted_block(voted_block, reject_txn_votes);
            certified_blocks.extend(blocks);
        }

        certified_blocks
    }

    fn add_voted_block(
        &mut self,
        voted_block: VerifiedBlock,
        reject_txn_votes: Vec<TransactionIndex>,
    ) -> Vec<CertifiedBlock> {
        if voted_block.round() <= self.gc_round {
            // Ignore the block and own votes, since they are outside of certifier GC bound.
            return vec![];
        }

        // Initialize the entry for the voted block.
        let vote_info = self.votes.entry(voted_block.reference()).or_default();
        if vote_info.block.is_some() {
            // Input block has already been processed and added to the state.
            return vec![];
        }
        vote_info.block = Some(voted_block.clone());
        vote_info.own_reject_txn_votes = reject_txn_votes;

        let mut certified_blocks = vec![];

        let now = self.context.clock.timestamp_utc_ms();

        // Update reject votes from the input block.
        for block_votes in voted_block.transaction_votes() {
            if block_votes.block_ref.round <= self.gc_round {
                // Block is outside of GC bound.
                continue;
            }
            let vote_info = self.votes.entry(block_votes.block_ref).or_default();
            for reject in &block_votes.rejects {
                vote_info
                    .reject_txn_votes
                    .entry(*reject)
                    .or_default()
                    .add_unique(voted_block.author(), &self.context.committee);
            }
            // Check if the target block is now certified after including the reject votes.
            // NOTE: votes can already exist for the target block and its transactions.
            if let Some(certified_block) = vote_info.take_certified_output(&self.context) {
                certified_blocks.push(certified_block);
            }
        }

        certified_blocks
    }

    fn add_proposed_block(&mut self, proposed_block: VerifiedBlock) -> Vec<CertifiedBlock> {
        if proposed_block.round() <= self.gc_round + 2 {
            // Skip if transactions that can be certified have already been GC'ed.
            // Skip also when the proposed block has been GC'ed from the certifier state.
            // This is possible because this function (add_proposed_block()) is async from
            // commit finalization, which advances the GC round of the certifier.
            return vec![];
        }
        debug!("Adding proposed block {}; gc round: {}", proposed_block.reference(), self.gc_round);

        if !self.votes.contains_key(&proposed_block.reference()) {
            debug!(
                "Proposed block {} not found in certifier state. GC round: {}",
                proposed_block.reference(),
                self.gc_round,
            );
            return vec![];
        }

        let now = self.context.clock.timestamp_utc_ms();

        // Certify transactions based on the accept votes from the proposed block's parents.
        // Some ancestor blocks may not be found, because either they have been GC'ed due to timing
        // issues or they were not recovered. It is ok to skip certifying blocks, which are best effort.
        let mut certified_blocks = vec![];
        for voting_ancestor in proposed_block.ancestors() {
            // Votes are limited to 1 round before the proposed block.
            if voting_ancestor.round + 1 != proposed_block.round() {
                continue;
            }
            let Some(voting_info) = self.votes.get(voting_ancestor) else {
                debug!(
                    "Proposed block {}: voting info not found for ancestor {}",
                    proposed_block.reference(),
                    voting_ancestor
                );
                continue;
            };
            let Some(voting_block) = voting_info.block.clone() else {
                debug!(
                    "Proposed block {}: voting block not found for ancestor {}",
                    proposed_block.reference(),
                    voting_ancestor
                );
                continue;
            };
            for target_ancestor in voting_block.ancestors() {
                // Target blocks are 1 round before the voting block.
                if target_ancestor.round + 1 != voting_block.round() {
                    continue;
                }
                let Some(target_vote_info) = self.votes.get_mut(target_ancestor) else {
                    debug!(
                        "Proposed block {}: target voting info not found for ancestor {}",
                        proposed_block.reference(),
                        target_ancestor
                    );
                    continue;
                };
                target_vote_info
                    .accept_block_votes
                    .add_unique(voting_block.author(), &self.context.committee);
                // Check if the target block is now certified after including the accept votes.
                if let Some(certified_block) = target_vote_info.take_certified_output(&self.context)
                {
                    certified_blocks.push(certified_block);
                }
            }
        }

        certified_blocks
    }

    /// Updates the GC round and cleans up obsolete internal state.
    fn update_gc_round(&mut self, gc_round: Round) {
        self.gc_round = gc_round;
        while let Some((block_ref, _)) = self.votes.first_key_value() {
            if block_ref.round <= self.gc_round {
                self.votes.pop_first();
            } else {
                break;
            }
        }
    }
}

/// VoteInfo keeps track of votes received for each transaction of this block,
/// possibly even before the block is received by this authority.
struct VoteInfo {
    // Content of the block.
    // None if the blocks has not been received.
    block: Option<VerifiedBlock>,
    // Rejection votes by this authority on this block.
    // This field is written when the block is first received and its transactions are voted on.
    // It is read from core after the block is accepted.
    own_reject_txn_votes: Vec<TransactionIndex>,
    // Accumulates implicit accept votes for the block and all transactions.
    accept_block_votes: StakeAggregator<QuorumThreshold>,
    // Accumulates reject votes per transaction in this block.
    reject_txn_votes: BTreeMap<TransactionIndex, StakeAggregator<QuorumThreshold>>,
    // Whether this block has been certified already.
    is_certified: bool,
}

impl VoteInfo {
    // If this block can now be certified, returns the output.
    // Otherwise, returns None.
    fn take_certified_output(&mut self, context: &Context) -> Option<CertifiedBlock> {
        let committee = &context.committee;
        if self.is_certified {
            // Skip if already certified.
            return None;
        }
        let Some(block) = self.block.as_ref() else {
            // Skip if the content of the block has not been received.
            return None;
        };

        if !self.accept_block_votes.reached_threshold(committee) {
            // Skip if the block is not certified.
            return None;
        }
        let mut rejected = vec![];
        for (idx, reject_txn_votes) in &self.reject_txn_votes {
            // The transaction is voted to be rejected.
            if reject_txn_votes.reached_threshold(committee) {
                rejected.push(*idx);
                continue;
            }
            // If a transaction does not have a quorum of accept votes minus the reject votes,
            // it is neither rejected nor certified. In this case the whole block cannot
            // be considered as certified.

            // accept_block_votes can be < reject_txn_votes on the transaction when reject_txn_votes
            // come from blocks more than 1 round higher, which do not add to the
            // accept votes of the block.
            //
            // Also, the total accept votes of a transactions is undercounted here.
            // If a block has accept votes from a quorum of authorities A, B and C, but one transaction
            // has a reject vote from D, the transaction and block are technically certified
            // and can be sent to fastpath. However, the computation here will not certify the transaction
            // or the block. This is still fine because the fastpath certification is optional.
            // The definite status of the transaction will be decided during post commit finalization.
            if self.accept_block_votes.stake().saturating_sub(reject_txn_votes.stake())
                < committee.quorum_threshold()
            {
                return None;
            }
        }
        // The block is certified.
        let accepted_txn_count = block.transactions().len().saturating_sub(rejected.len());
        tracing::trace!(
            "Certified block {} accepted tx count: {accepted_txn_count} & rejected txn count: {}",
            block.reference(),
            rejected.len()
        );

        self.is_certified = true;
        Some(CertifiedBlock { block: block.clone(), rejected })
    }
}

impl Default for VoteInfo {
    fn default() -> Self {
        Self {
            block: None,
            own_reject_txn_votes: vec![],
            accept_block_votes: StakeAggregator::new(),
            reject_txn_votes: BTreeMap::new(),
            is_certified: false,
        }
    }
}
