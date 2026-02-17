// Portions of this file are derived from Mysticeti consensus (MystenLabs/sui).
// Original source: https://github.com/MystenLabs/sui/tree/main/consensus/core/src/block_verifier.rs
// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::transaction::TransactionVerifier;
use bytes::Bytes;
use std::{collections::BTreeSet, sync::Arc};
use types::consensus::block::{BlockRef, TransactionIndex};
use types::consensus::{
    block::{BlockAPI, GENESIS_ROUND, SignedBlock, VerifiedBlock, genesis_blocks},
    context::Context,
};
use types::error::{ConsensusError, ConsensusResult};

pub trait BlockVerifier: Send + Sync + 'static {
    /// Verifies a block and its transactions, checking signatures, size limits,
    /// and transaction validity. All honest validators should produce the same verification
    /// outcome for the same block, so any verification error should be due to equivocation.
    /// Returns the verified block.
    ///
    /// When Mysticeti fastpath is enabled, it also votes on the transactions in verified blocks,
    /// and can return a non-empty list of rejected transaction indices. Different honest
    /// validators may vote differently on transactions.
    ///
    /// The method takes both the SignedBlock and its serialized bytes, to avoid re-serializing the block.
    #[allow(private_interfaces)]
    fn verify_and_vote(
        &self,
        block: SignedBlock,
        serialized_block: Bytes,
    ) -> ConsensusResult<(VerifiedBlock, Vec<TransactionIndex>)>;

    /// Votes on the transactions in a verified block.
    /// This is used to vote on transactions in a verified block, without having to verify the block again. The method
    /// will verify the transactions and vote on them.
    fn vote(&self, block: &VerifiedBlock) -> ConsensusResult<Vec<TransactionIndex>>;
}

/// `SignedBlockVerifier` checks the validity of a block.
///
/// Blocks that fail verification at one honest authority will be rejected by all other honest
/// authorities as well. The means invalid blocks, and blocks with an invalid ancestor, will never
/// be accepted into the DAG.
pub(crate) struct SignedBlockVerifier {
    context: Arc<Context>,
    genesis: BTreeSet<BlockRef>,
    transaction_verifier: Arc<dyn TransactionVerifier>,
}

impl SignedBlockVerifier {
    pub(crate) fn new(
        context: Arc<Context>,
        transaction_verifier: Arc<dyn TransactionVerifier>,
    ) -> Self {
        let genesis = genesis_blocks(&context).into_iter().map(|b| b.reference()).collect();
        Self { context, genesis, transaction_verifier }
    }

    fn verify_block(&self, block: &SignedBlock) -> ConsensusResult<()> {
        let committee = &self.context.committee;
        // The block must belong to the current epoch and have valid authority index,
        // before having its signature verified.
        if block.epoch() != committee.epoch() {
            return Err(ConsensusError::WrongEpoch {
                expected: committee.epoch(),
                actual: block.epoch(),
            });
        }
        if block.round() == 0 {
            return Err(ConsensusError::UnexpectedGenesisBlock);
        }
        if !committee.is_valid_index(block.author()) {
            return Err(ConsensusError::InvalidAuthorityIndex {
                index: block.author(),
                max: committee.size() - 1,
            });
        }

        // Verify the block's signature.
        block.verify_signature(&self.context)?;

        // Verify the block's ancestor refs are consistent with the block's round,
        // and total parent stakes reach quorum.
        if block.ancestors().len() > committee.size() {
            return Err(ConsensusError::TooManyAncestors(
                block.ancestors().len(),
                committee.size(),
            ));
        }
        if block.ancestors().is_empty() {
            return Err(ConsensusError::InsufficientParentStakes {
                parent_stakes: 0,
                quorum: committee.quorum_threshold(),
            });
        }
        let mut seen_ancestors = vec![false; committee.size()];
        let mut parent_stakes = 0;
        for (i, ancestor) in block.ancestors().iter().enumerate() {
            if !committee.is_valid_index(ancestor.author) {
                return Err(ConsensusError::InvalidAuthorityIndex {
                    index: ancestor.author,
                    max: committee.size() - 1,
                });
            }
            if (i == 0 && ancestor.author != block.author())
                || (i > 0 && ancestor.author == block.author())
            {
                return Err(ConsensusError::InvalidAncestorPosition {
                    block_authority: block.author(),
                    ancestor_authority: ancestor.author,
                    position: i,
                });
            }
            if ancestor.round >= block.round() {
                return Err(ConsensusError::InvalidAncestorRound {
                    ancestor: ancestor.round,
                    block: block.round(),
                });
            }
            if ancestor.round == GENESIS_ROUND && !self.genesis.contains(ancestor) {
                return Err(ConsensusError::InvalidGenesisAncestor(*ancestor));
            }
            if seen_ancestors[ancestor.author] {
                return Err(ConsensusError::DuplicatedAncestorsAuthority(ancestor.author));
            }
            seen_ancestors[ancestor.author] = true;
            // Block must have round >= 1 so checked_sub(1) should be safe.
            if ancestor.round == block.round().checked_sub(1).unwrap() {
                parent_stakes += committee.stake_by_index(ancestor.author);
            }
        }
        if !committee.reached_quorum(parent_stakes) {
            return Err(ConsensusError::InsufficientParentStakes {
                parent_stakes,
                quorum: committee.quorum_threshold(),
            });
        }

        let batch: Vec<_> = block.transactions().iter().map(|t| t.data()).collect();

        self.check_transactions(&batch)
    }

    pub(crate) fn check_transactions(&self, batch: &[&[u8]]) -> ConsensusResult<()> {
        let max_transaction_size_limit =
            self.context.protocol_config.max_transaction_size_bytes() as usize;
        for t in batch {
            if t.len() > max_transaction_size_limit && max_transaction_size_limit > 0 {
                return Err(ConsensusError::TransactionTooLarge {
                    size: t.len(),
                    limit: max_transaction_size_limit,
                });
            }
        }

        let max_num_transactions_limit =
            self.context.protocol_config.max_num_transactions_in_block() as usize;
        if batch.len() > max_num_transactions_limit && max_num_transactions_limit > 0 {
            return Err(ConsensusError::TooManyTransactions {
                count: batch.len(),
                limit: max_num_transactions_limit,
            });
        }

        let total_transactions_size_limit =
            self.context.protocol_config.max_transactions_in_block_bytes() as usize;
        if batch.iter().map(|t| t.len()).sum::<usize>() > total_transactions_size_limit
            && total_transactions_size_limit > 0
        {
            return Err(ConsensusError::TooManyTransactionBytes {
                size: batch.len(),
                limit: total_transactions_size_limit,
            });
        }
        Ok(())
    }
}

// All block verification logic are implemented below.
impl BlockVerifier for SignedBlockVerifier {
    fn verify_and_vote(
        &self,
        block: SignedBlock,
        serialized_block: Bytes,
    ) -> ConsensusResult<(VerifiedBlock, Vec<TransactionIndex>)> {
        self.verify_block(&block)?;

        // If the block verification passed then we can produce the verified block, but we should only return it if the transaction verification passed as well.
        let verified_block = VerifiedBlock::new_verified(block, serialized_block);

        let rejected_transactions = self.vote(&verified_block)?;
        Ok((verified_block, rejected_transactions))
    }

    fn vote(&self, block: &VerifiedBlock) -> ConsensusResult<Vec<TransactionIndex>> {
        self.transaction_verifier
            .verify_and_vote_batch(&block.reference(), &block.transactions_data())
            .map_err(|e| ConsensusError::InvalidTransaction(e.to_string()))
    }
}

/// Allows all transactions to pass verification, for testing.
pub struct NoopBlockVerifier;

impl BlockVerifier for NoopBlockVerifier {
    #[allow(private_interfaces)]
    fn verify_and_vote(
        &self,
        _block: SignedBlock,
        _serialized_block: Bytes,
    ) -> ConsensusResult<(VerifiedBlock, Vec<TransactionIndex>)> {
        Ok((VerifiedBlock::new_verified(_block, _serialized_block), vec![]))
    }

    fn vote(&self, _block: &VerifiedBlock) -> ConsensusResult<Vec<TransactionIndex>> {
        Ok(vec![])
    }
}
