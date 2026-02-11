use parking_lot::Mutex;
use std::{collections::BTreeMap, sync::Arc};
use tap::TapFallible;
use thiserror::Error;
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tokio::sync::oneshot;
use tracing::{debug, error, warn};
use types::committee::Epoch;
use types::consensus::{
    block::{
        BlockRef, NUM_RESERVED_TRANSACTION_INDICES, PING_TRANSACTION_INDEX, Round, Transaction,
        TransactionIndex,
    },
    context::Context,
};

/// The maximum number of transactions pending to the queue to be pulled for block proposal
const MAX_PENDING_TRANSACTIONS: usize = 2_000;

/// The guard acts as an acknowledgment mechanism for the inclusion of the transactions to a block.
/// When its last transaction is included to a block then `included_in_block_ack` will be signalled.
/// If the guard is dropped without getting acknowledged that means the transactions have not been
/// included to a block and the consensus is shutting down.
pub struct TransactionsGuard {
    // Holds a list of transactions to be included in the block.
    // A TransactionsGuard may be partially consumed by `TransactionConsumer`, in which case, this holds the remaining transactions.
    transactions: Vec<Transaction>,

    // When the transactions are included in a block, this will be signalled with
    // the following information
    included_in_block_ack: oneshot::Sender<(
        // The block reference in which the transactions have been included
        BlockRef,
        // The indices of the transactions that have been included in the block
        Vec<TransactionIndex>,
        // A receiver to notify the submitter about the block status
        oneshot::Receiver<BlockStatus>,
    )>,
}

/// The TransactionConsumer is responsible for fetching the next transactions to be included for the block proposals.
/// The transactions are submitted to a channel which is shared between the TransactionConsumer and the TransactionClient
/// and are pulled every time the `next` method is called.
pub struct TransactionConsumer {
    tx_receiver: Receiver<TransactionsGuard>,
    max_transactions_in_block_bytes: u64,
    max_num_transactions_in_block: u64,
    pending_transactions: Option<TransactionsGuard>,
    block_status_subscribers: Arc<Mutex<BTreeMap<BlockRef, Vec<oneshot::Sender<BlockStatus>>>>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
#[allow(unused)]
pub enum BlockStatus {
    /// The block has been sequenced as part of a committed sub dag. That means that any transaction that has been included in the block
    /// has been committed as well.
    Sequenced(BlockRef),
    /// The block has been garbage collected and will never be committed. Any transactions that have been included in the block should also
    /// be considered as impossible to be committed as part of this block and might need to be retried
    GarbageCollected(BlockRef),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum LimitReached {
    // The maximum number of transactions have been included
    MaxNumOfTransactions,
    // The maximum number of bytes have been included
    MaxBytes,
    // All available transactions have been included
    AllTransactionsIncluded,
}

impl TransactionConsumer {
    pub fn new(tx_receiver: Receiver<TransactionsGuard>, context: Arc<Context>) -> Self {
        // max_num_transactions_in_block - 1 is the max possible transaction index in a block.
        // TransactionIndex::MAX is reserved for the ping transaction.
        // Indexes down to TransactionIndex::MAX - 8 are also reserved for future use.
        // This check makes sure they do not overlap.
        assert!(
            context.protocol_config.max_num_transactions_in_block().saturating_sub(1)
                < TransactionIndex::MAX.saturating_sub(NUM_RESERVED_TRANSACTION_INDICES) as u64,
            "Unsupported max_num_transactions_in_block: {}",
            context.protocol_config.max_num_transactions_in_block()
        );

        Self {
            tx_receiver,
            max_transactions_in_block_bytes: context
                .protocol_config
                .max_transactions_in_block_bytes(),
            max_num_transactions_in_block: context.protocol_config.max_num_transactions_in_block(),
            pending_transactions: None,
            block_status_subscribers: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    // Attempts to fetch the next transactions that have been submitted for sequence. Respects the `max_transactions_in_block_bytes`
    // and `max_num_transactions_in_block` parameters specified via protocol config.
    // This returns one or more transactions to be included in the block and a callback to acknowledge the inclusion of those transactions.
    // Also returns a `LimitReached` enum to indicate which limit type has been reached.
    #[allow(clippy::type_complexity)]
    pub fn next(&mut self) -> (Vec<Transaction>, Box<dyn FnOnce(BlockRef)>, LimitReached) {
        let mut transactions = Vec::new();
        let mut acks = Vec::new();
        let mut total_bytes = 0;
        let mut limit_reached = LimitReached::AllTransactionsIncluded;

        // Handle one batch of incoming transactions from TransactionGuard.
        // The method will return `None` if all the transactions can be included in the block. Otherwise none of the transactions will be
        // included in the block and the method will return the TransactionGuard.
        let mut handle_txs = |t: TransactionsGuard| -> Option<TransactionsGuard> {
            // If no transactions are submitted, it means that the transaction guard represents a ping transaction.
            // In this case, we need to push the `PING_TRANSACTION_INDEX` to the indices vector.
            let transactions_num = t.transactions.len() as u64;
            if transactions_num == 0 {
                acks.push((t.included_in_block_ack, vec![PING_TRANSACTION_INDEX]));
                return None;
            }

            // Check if the total bytes of the transactions exceed the max transactions in block bytes.
            let transactions_bytes =
                t.transactions.iter().map(|t| t.data().len()).sum::<usize>() as u64;
            if total_bytes + transactions_bytes > self.max_transactions_in_block_bytes {
                limit_reached = LimitReached::MaxBytes;
                return Some(t);
            }
            if transactions.len() as u64 + transactions_num > self.max_num_transactions_in_block {
                limit_reached = LimitReached::MaxNumOfTransactions;
                return Some(t);
            }

            total_bytes += transactions_bytes;

            // Calculate indices for this batch
            let start_idx = transactions.len() as TransactionIndex;
            let indices: Vec<TransactionIndex> =
                (start_idx..start_idx + t.transactions.len() as TransactionIndex).collect();

            // The transactions can be consumed, register its ack and transaction
            // indices to be sent with the ack.
            acks.push((t.included_in_block_ack, indices));
            transactions.extend(t.transactions);
            None
        };

        if let Some(t) = self.pending_transactions.take()
            && let Some(pending_transactions) = handle_txs(t)
        {
            debug!(
                "Previously pending transaction(s) should fit into an empty block! Dropping: {:?}",
                pending_transactions.transactions
            );
        }

        // Until we have reached the limit for the pull.
        // We may have already reached limit in the first iteration above, in which case we stop immediately.
        while self.pending_transactions.is_none() {
            if let Ok(t) = self.tx_receiver.try_recv() {
                self.pending_transactions = handle_txs(t);
            } else {
                break;
            }
        }

        let block_status_subscribers = self.block_status_subscribers.clone();
        (
            transactions,
            Box::new(move |block_ref: BlockRef| {
                let mut block_status_subscribers = block_status_subscribers.lock();

                for (ack, tx_indices) in acks {
                    let (status_tx, status_rx) = oneshot::channel();

                    block_status_subscribers.entry(block_ref).or_default().push(status_tx);

                    let _ = ack.send((block_ref, tx_indices, status_rx));
                }
            }),
            limit_reached,
        )
    }

    /// Notifies all the transaction submitters who are waiting to receive an update on the status of the block.
    /// The `committed_blocks` are the blocks that have been committed and the `gc_round` is the round up to which the blocks have been garbage collected.
    /// First we'll notify for all the committed blocks, and then for all the blocks that have been garbage collected.
    pub fn notify_own_blocks_status(&self, committed_blocks: Vec<BlockRef>, gc_round: Round) {
        // Notify for all the committed blocks first
        let mut block_status_subscribers = self.block_status_subscribers.lock();
        for block_ref in committed_blocks {
            if let Some(subscribers) = block_status_subscribers.remove(&block_ref) {
                subscribers.into_iter().for_each(|s| {
                    let _ = s.send(BlockStatus::Sequenced(block_ref));
                });
            }
        }

        // Now notify everyone <= gc_round that their block has been garbage collected and clean up the entries
        while let Some((block_ref, subscribers)) = block_status_subscribers.pop_first() {
            if block_ref.round <= gc_round {
                subscribers.into_iter().for_each(|s| {
                    let _ = s.send(BlockStatus::GarbageCollected(block_ref));
                });
            } else {
                block_status_subscribers.insert(block_ref, subscribers);
                break;
            }
        }
    }

    #[cfg(test)]
    pub fn subscribe_for_block_status_testing(
        &self,
        block_ref: BlockRef,
    ) -> oneshot::Receiver<BlockStatus> {
        let (tx, rx) = oneshot::channel();
        let mut block_status_subscribers = self.block_status_subscribers.lock();
        block_status_subscribers.entry(block_ref).or_default().push(tx);
        rx
    }

    #[cfg(test)]
    fn is_empty(&mut self) -> bool {
        if self.pending_transactions.is_some() {
            return false;
        }
        if let Ok(t) = self.tx_receiver.try_recv() {
            self.pending_transactions = Some(t);
            return false;
        }
        true
    }
}

#[derive(Clone)]
pub struct TransactionClient {
    context: Arc<Context>,
    sender: Sender<TransactionsGuard>,
    max_transaction_size: u64,
    max_transactions_in_block_bytes: u64,
    max_transactions_in_block_count: u64,
}

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("Failed to submit transaction, consensus is shutting down: {0}")]
    ConsensusShuttingDown(String),

    #[error("Transaction size ({0}B) is over limit ({1}B)")]
    OversizedTransaction(u64, u64),

    #[error("Transaction bundle size ({0}B) is over limit ({1}B)")]
    OversizedTransactionBundleBytes(u64, u64),

    #[error("Transaction bundle count ({0}) is over limit ({1})")]
    OversizedTransactionBundleCount(u64, u64),
}

impl TransactionClient {
    pub fn new(context: Arc<Context>) -> (Self, Receiver<TransactionsGuard>) {
        Self::new_with_max_pending_transactions(context, MAX_PENDING_TRANSACTIONS)
    }

    fn new_with_max_pending_transactions(
        context: Arc<Context>,
        max_pending_transactions: usize,
    ) -> (Self, Receiver<TransactionsGuard>) {
        let (sender, receiver) = channel(max_pending_transactions);
        (
            Self {
                sender,
                max_transaction_size: context.protocol_config.max_transaction_size_bytes(),

                max_transactions_in_block_bytes: context
                    .protocol_config
                    .max_transactions_in_block_bytes(),
                max_transactions_in_block_count: context
                    .protocol_config
                    .max_num_transactions_in_block(),
                context: context.clone(),
            },
            receiver,
        )
    }

    /// Returns the current epoch of this client.
    pub fn epoch(&self) -> Epoch {
        self.context.committee.epoch()
    }

    /// Submits a list of transactions to be sequenced. The method returns when all the transactions have been successfully included
    /// to next proposed blocks.
    ///
    /// If `transactions` is empty, then this will be interpreted as a "ping" signal from the client in order to get information about the next
    /// block and simulate a transaction inclusion to the next block. In this an empty vector of the transaction index will be returned as response
    /// and the block status receiver.
    pub async fn submit(
        &self,
        transactions: Vec<Vec<u8>>,
    ) -> Result<(BlockRef, Vec<TransactionIndex>, oneshot::Receiver<BlockStatus>), ClientError>
    {
        let included_in_block = self.submit_no_wait(transactions).await?;
        included_in_block
            .await
            .tap_err(|e| warn!("Transaction acknowledge failed with {:?}", e))
            .map_err(|e| ClientError::ConsensusShuttingDown(e.to_string()))
    }

    /// Submits a list of transactions to be sequenced.
    /// If any transaction's length exceeds `max_transaction_size`, no transaction will be submitted.
    /// That shouldn't be the common case as sizes should be aligned between consensus and client. The method returns
    /// a receiver to wait on until the transactions has been included in the next block to get proposed. The consumer should
    /// wait on it to consider as inclusion acknowledgement. If the receiver errors then consensus is shutting down and transaction
    /// has not been included to any block.
    /// If multiple transactions are submitted, the method will attempt to bundle them together in a single block. If the total size of
    /// the transactions exceeds `max_transactions_in_block_bytes`, no transaction will be submitted and an error will be returned instead.
    /// Similar if transactions exceed `max_transactions_in_block_count` an error will be returned.
    pub async fn submit_no_wait(
        &self,
        transactions: Vec<Vec<u8>>,
    ) -> Result<
        oneshot::Receiver<(BlockRef, Vec<TransactionIndex>, oneshot::Receiver<BlockStatus>)>,
        ClientError,
    > {
        let (included_in_block_ack_send, included_in_block_ack_receive) = oneshot::channel();

        let mut bundle_size = 0;

        if transactions.len() as u64 > self.max_transactions_in_block_count {
            return Err(ClientError::OversizedTransactionBundleCount(
                transactions.len() as u64,
                self.max_transactions_in_block_count,
            ));
        }

        for transaction in &transactions {
            if transaction.len() as u64 > self.max_transaction_size {
                return Err(ClientError::OversizedTransaction(
                    transaction.len() as u64,
                    self.max_transaction_size,
                ));
            }
            bundle_size += transaction.len() as u64;

            if bundle_size > self.max_transactions_in_block_bytes {
                return Err(ClientError::OversizedTransactionBundleBytes(
                    bundle_size,
                    self.max_transactions_in_block_bytes,
                ));
            }
        }

        let t = TransactionsGuard {
            transactions: transactions.into_iter().map(Transaction::new).collect(),
            included_in_block_ack: included_in_block_ack_send,
        };
        self.sender
            .send(t)
            .await
            .tap_err(|e| error!("Submit transactions failed with {:?}", e))
            .map_err(|e| ClientError::ConsensusShuttingDown(e.to_string()))?;
        Ok(included_in_block_ack_receive)
    }
}

/// `TransactionVerifier` implementation is supplied by Sui to validate transactions in a block,
/// before acceptance of the block.
pub trait TransactionVerifier: Send + Sync + 'static {
    /// Determines if this batch of transactions is valid.
    /// Fails if any one of the transactions is invalid.
    fn verify_batch(&self, batch: &[&[u8]]) -> Result<(), ValidationError>;

    /// Returns indices of transactions to reject, or a transaction validation error.
    /// Currently only uncertified user transactions can be voted to reject, which are created
    /// by Mysticeti fastpath client.
    /// Honest validators may disagree on voting for uncertified user transactions.
    /// The other types of transactions are implicitly voted to be accepted if they pass validation.
    ///
    /// Honest validators should produce the same validation outcome on the same batch of
    /// transactions. So if a batch from a peer fails validation, the peer is equivocating.
    fn verify_and_vote_batch(
        &self,
        block_ref: &BlockRef,
        batch: &[&[u8]],
    ) -> Result<Vec<TransactionIndex>, ValidationError>;
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),
}

/// `NoopTransactionVerifier` accepts all transactions.
#[cfg(any(test, msim))]
pub struct NoopTransactionVerifier;

#[cfg(any(test, msim))]
impl TransactionVerifier for NoopTransactionVerifier {
    fn verify_batch(&self, _batch: &[&[u8]]) -> Result<(), ValidationError> {
        Ok(())
    }

    fn verify_and_vote_batch(
        &self,
        _block_ref: &BlockRef,
        _batch: &[&[u8]],
    ) -> Result<Vec<TransactionIndex>, ValidationError> {
        Ok(vec![])
    }
}
