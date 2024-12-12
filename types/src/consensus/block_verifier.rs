use std::{collections::BTreeSet, sync::Arc};

use super::block::{
    genesis_blocks, BlockAPI as _, BlockRef, BlockTimestampMs, SignedBlock, VerifiedBlock,
    GENESIS_ROUND,
};
use super::context::Context;
use super::transaction::TransactionVerifier;
use crate::accumulator::{self, AccumulatorStore};
use fastcrypto::hash::MultisetHash;
use tokio::task::watch::error;
use tracing::{error, info};

use crate::error::{ConsensusError, ConsensusResult};

pub trait BlockVerifier: Send + Sync + 'static {
    /// Verifies a block's metadata and transactions.
    /// This is called before examining a block's causal history.
    fn verify(&self, block: &SignedBlock) -> ConsensusResult<()>;

    /// Verifies a block w.r.t. ancestor blocks.
    /// This is called after a block has complete causal history locally,
    /// and is ready to be accepted into the DAG.
    ///
    /// Caller must make sure ancestors corresponse to block.ancestors() 1-to-1, in the same order.
    fn check_ancestors(
        &self,
        block: &VerifiedBlock,
        ancestors: &[VerifiedBlock],
    ) -> ConsensusResult<()>;
}

/// `SignedBlockVerifier` checks the validity of a block.
///
/// Blocks that fail verification at one honest authority will be rejected by all other honest
/// authorities as well. The means invalid blocks, and blocks with an invalid ancestor, will never
/// be accepted into the DAG.
pub struct SignedBlockVerifier {
    context: Arc<Context>,
    genesis: BTreeSet<BlockRef>,
    transaction_verifier: Arc<dyn TransactionVerifier>,
    accumulator_store: Arc<dyn AccumulatorStore>,
}

impl SignedBlockVerifier {
    pub fn new(
        context: Arc<Context>,
        transaction_verifier: Arc<dyn TransactionVerifier>,
        accumulator_store: Arc<dyn AccumulatorStore>,
    ) -> Self {
        let genesis = genesis_blocks(context.clone())
            .into_iter()
            .map(|b| b.reference())
            .collect();
        Self {
            context,
            genesis,
            transaction_verifier,
            accumulator_store,
        }
    }
}

// All block verification logic are implemented below.
impl BlockVerifier for SignedBlockVerifier {
    fn verify(&self, block: &SignedBlock) -> ConsensusResult<()> {
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

        // If the block contains a state commit, verify the accumulator.
        if let Some(state_commit) = block.state_commit() {
            let accumulator = self
                .accumulator_store
                .get_root_state_accumulator_for_commit(state_commit.commit());
            if let Ok(Some(stored_state_hash)) = accumulator {
                if stored_state_hash != *state_commit.state_hash() {
                    error!(
                        "State hash mismatch: expected {:?}, actual {:?}",
                        stored_state_hash.digest(),
                        state_commit.state_hash().digest()
                    );
                    return Err(ConsensusError::InvalidStateHash {
                        expected: stored_state_hash.digest(),
                        actual: state_commit.state_hash().digest(),
                    });
                } else {
                    info!(
                        "State hash matches the stored state hash. {:?}",
                        stored_state_hash.digest()
                    );
                }
            } else {
                info!("State hash not found in the accumulator store.");
            }
        }

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
                return Err(ConsensusError::DuplicatedAncestorsAuthority(
                    ancestor.author,
                ));
            }
            seen_ancestors[ancestor.author] = true;
            // Block must have round >= 1 so checked_sub(1) should be safe.
            if ancestor.round == block.round().checked_sub(1).unwrap() {
                parent_stakes += committee.stake(ancestor.author);
            }
        }
        if !committee.reached_quorum(parent_stakes) {
            return Err(ConsensusError::InsufficientParentStakes {
                parent_stakes,
                quorum: committee.quorum_threshold(),
            });
        }

        // TODO: check transaction size, total size and count.
        let batch: Vec<_> = block.transactions().iter().map(|t| t.data()).collect();
        self.transaction_verifier
            .verify_batch(&batch)
            .map_err(|e| ConsensusError::InvalidTransaction(format!("{e:?}")))
    }

    fn check_ancestors(
        &self,
        block: &VerifiedBlock,
        ancestors: &[VerifiedBlock],
    ) -> ConsensusResult<()> {
        assert_eq!(block.ancestors().len(), ancestors.len());
        // This checks the invariant that block timestamp >= max ancestor timestamp.
        let mut max_timestamp_ms = BlockTimestampMs::MIN;
        for (ancestor_ref, ancestor_block) in block.ancestors().iter().zip(ancestors.iter()) {
            assert_eq!(ancestor_ref, &ancestor_block.reference());
            max_timestamp_ms = max_timestamp_ms.max(ancestor_block.timestamp_ms());
        }
        if max_timestamp_ms > block.timestamp_ms() {
            return Err(ConsensusError::InvalidBlockTimestamp {
                max_timestamp_ms,
                block_timestamp_ms: block.timestamp_ms(),
            });
        }
        Ok(())
    }
}

pub struct NoopBlockVerifier;

impl BlockVerifier for NoopBlockVerifier {
    fn verify(&self, _block: &SignedBlock) -> ConsensusResult<()> {
        Ok(())
    }

    fn check_ancestors(
        &self,
        _block: &VerifiedBlock,
        _ancestors: &[VerifiedBlock],
    ) -> ConsensusResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use futures::{stream::FuturesUnordered, StreamExt};

    use tokio::time::timeout;

    use super::super::{
        block::BlockRef,
        context::Context,
        transaction::{TransactionClient, TransactionConsumer},
    };

    use crate::parameters::Parameters;

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn basic_submit_and_consume() {
        let context = Arc::new(Context::new_for_test(4).0);
        let (client, tx_receiver) = TransactionClient::new(context.clone());
        let mut consumer = TransactionConsumer::new(tx_receiver, context.clone(), None);

        // submit asynchronously the transactions and keep the waiters
        let mut included_in_block_waiters = FuturesUnordered::new();
        for i in 0..3 {
            let transaction =
                bcs::to_bytes(&format!("transaction {i}")).expect("Serialization should not fail.");
            let w = client
                .submit_no_wait(vec![transaction])
                .await
                .expect("Shouldn't submit successfully transaction");
            included_in_block_waiters.push(w);
        }

        // now pull the transactions from the consumer
        let (transactions, ack_transactions) = consumer.next();
        assert_eq!(transactions.len(), 3);

        for (i, t) in transactions.iter().enumerate() {
            let t: String = bcs::from_bytes(t.data()).unwrap();
            assert_eq!(format!("transaction {i}").to_string(), t);
        }

        assert!(
            timeout(Duration::from_secs(1), included_in_block_waiters.next())
                .await
                .is_err(),
            "We should expect to timeout as none of the transactions have been acknowledged yet"
        );

        // Now acknowledge the inclusion of transactions
        ack_transactions(BlockRef::MIN);

        // Now make sure that all the waiters have returned
        while let Some(result) = included_in_block_waiters.next().await {
            assert!(result.is_ok());
        }

        // try to pull again transactions, result should be empty
        assert!(consumer.is_empty());
    }

    #[tokio::test]
    async fn submit_over_max_fetch_size_and_consume() {
        let context = Arc::new(Context::new_for_test(4).0.with_parameters(Parameters {
            consensus_max_transactions_in_block_bytes: 100,
            consensus_max_transaction_size_bytes: 100,
            ..Default::default()
        }));
        let (client, tx_receiver) = TransactionClient::new(context.clone());
        let mut consumer = TransactionConsumer::new(tx_receiver, context.clone(), None);

        // submit some transactions
        for i in 0..10 {
            let transaction =
                bcs::to_bytes(&format!("transaction {i}")).expect("Serialization should not fail.");
            let _w = client
                .submit_no_wait(vec![transaction])
                .await
                .expect("Shouldn't submit successfully transaction");
        }

        // now pull the transactions from the consumer
        let mut all_transactions = Vec::new();
        let (transactions, _ack_transactions) = consumer.next();
        assert_eq!(transactions.len(), 7);

        // ensure their total size is less than `max_bytes_to_fetch`
        let total_size: u64 = transactions.iter().map(|t| t.data().len() as u64).sum();
        assert!(
            total_size <= context.parameters.consensus_max_transactions_in_block_bytes,
            "Should have fetched transactions up to {}",
            context.parameters.consensus_max_transactions_in_block_bytes
        );
        all_transactions.extend(transactions);

        // try to pull again transactions, next should be provided
        let (transactions, _ack_transactions) = consumer.next();
        assert_eq!(transactions.len(), 3);

        // ensure their total size is less than `max_bytes_to_fetch`
        let total_size: u64 = transactions.iter().map(|t| t.data().len() as u64).sum();
        assert!(
            total_size <= context.parameters.consensus_max_transactions_in_block_bytes,
            "Should have fetched transactions up to {}",
            context.parameters.consensus_max_transactions_in_block_bytes
        );
        all_transactions.extend(transactions);

        // try to pull again transactions, result should be empty
        assert!(consumer.is_empty());

        for (i, t) in all_transactions.iter().enumerate() {
            let t: String = bcs::from_bytes(t.data()).unwrap();
            assert_eq!(format!("transaction {i}").to_string(), t);
        }
    }

    #[tokio::test]
    async fn submit_large_batch_and_ack() {
        let context = Arc::new(Context::new_for_test(4).0.with_parameters(Parameters {
            consensus_max_transactions_in_block_bytes: 100,
            consensus_max_transaction_size_bytes: 100,
            ..Default::default()
        }));
        let (client, tx_receiver) = TransactionClient::new(context.clone());
        let mut consumer = TransactionConsumer::new(tx_receiver, context.clone(), None);
        let mut all_receivers = Vec::new();
        // submit a few transactions individually.
        for i in 0..10 {
            let transaction =
                bcs::to_bytes(&format!("transaction {i}")).expect("Serialization should not fail.");
            let w = client
                .submit_no_wait(vec![transaction])
                .await
                .expect("Shouldn't submit successfully transaction");
            all_receivers.push(w);
        }

        // construct a over-size-limit batch and submit, which should get broken into smaller ones.
        {
            let transactions: Vec<_> = (10..32)
                .map(|i| {
                    bcs::to_bytes(&format!("transaction {i}"))
                        .expect("Serialization should not fail.")
                })
                .collect();
            let w = client
                .submit_no_wait(transactions)
                .await
                .expect("Shouldn't submit successfully transaction");
            all_receivers.push(w);
        }

        // submit another individual transaction.
        {
            let i = 32;
            let transaction =
                bcs::to_bytes(&format!("transaction {i}")).expect("Serialization should not fail.");
            let w = client
                .submit_no_wait(vec![transaction])
                .await
                .expect("Shouldn't submit successfully transaction");
            all_receivers.push(w);
        }

        // now pull the transactions from the consumer.
        // we expect all transactions are fetched in order, not missing any, and not exceeding the size limit.
        let mut all_transactions = Vec::new();
        let mut all_acks: Vec<Box<dyn FnOnce(BlockRef)>> = Vec::new();
        while !consumer.is_empty() {
            let (transactions, ack_transactions) = consumer.next();

            let total_size: u64 = transactions.iter().map(|t| t.data().len() as u64).sum();
            assert!(
                total_size <= context.parameters.consensus_max_transactions_in_block_bytes,
                "Should have fetched transactions up to {}",
                context.parameters.consensus_max_transactions_in_block_bytes
            );

            all_transactions.extend(transactions);
            all_acks.push(ack_transactions);
        }

        // verify the number of transactions as well as the content.
        assert_eq!(all_transactions.len(), 33);
        for (i, t) in all_transactions.iter().enumerate() {
            let t: String = bcs::from_bytes(t.data()).unwrap();
            assert_eq!(format!("transaction {i}").to_string(), t);
        }

        // now acknowledge the inclusion of all transactions.
        for ack in all_acks {
            ack(BlockRef::MIN);
        }

        // expect all receivers to be resolved.
        for w in all_receivers {
            assert!(w.await.is_ok());
        }
    }
}
