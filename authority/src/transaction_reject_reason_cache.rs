use std::collections::BTreeMap;

use parking_lot::RwLock;
use tracing::trace;
use types::committee::AuthorityIndex;
use types::committee::EpochId;
use types::consensus::ConsensusPosition;
use types::consensus::block::{BlockDigest, BlockRef, TransactionIndex};
use types::error::SomaError;

use crate::consensus_tx_status_cache::CONSENSUS_STATUS_RETENTION_ROUNDS;

#[cfg(test)]
use types::consensus::block::Round;

/// A cache that maintains rejection reasons (SomaError) when validators cast reject votes for transactions
/// during the Mysticeti consensus fast path voting process.
///
/// This cache serves as a bridge between the consensus voting mechanism and client-facing APIs,
/// allowing detailed error information to be returned when querying transaction status.
///
/// ## Key Characteristics:
/// - **Mysticeti Fast Path Only**: Only populated when transactions are voted on via the mysticeti
///   fast path, as it relies on consensus position (epoch, block, index) to uniquely identify transactions
/// - **Pre-consensus Rejections**: Direct rejections during transaction submission (before consensus
///   propagation) are not cached since these transactions never enter the consensus pipeline
/// - **Automatic Cleanup**: Maintains a retention period based on the last committed leader round
///   and automatically purges older entries to prevent unbounded memory growth
///
/// ## Use Cases:
/// - Providing detailed rejection reasons to clients querying transaction status
/// - Debugging transaction failures in the fast path voting process
pub(crate) struct TransactionRejectReasonCache {
    cache: RwLock<BTreeMap<ConsensusPosition, SomaError>>,
    retention_rounds: u32,
    epoch: EpochId,
}

impl TransactionRejectReasonCache {
    pub fn new(retention_rounds: Option<u32>, epoch: EpochId) -> Self {
        Self {
            cache: Default::default(),
            retention_rounds: retention_rounds.unwrap_or(CONSENSUS_STATUS_RETENTION_ROUNDS),
            epoch,
        }
    }

    /// Records a rejection vote reason for a transaction at the specified consensus position. The consensus `position` that
    /// uniquely identifies the transaction and the `reason` (SomaError) that caused the transaction to be rejected during voting
    /// should be provided.
    pub fn set_rejection_vote_reason(&self, position: ConsensusPosition, reason: &SomaError) {
        debug_assert_eq!(position.epoch, self.epoch, "Epoch mismatch");
        self.cache.write().insert(position, reason.clone());
    }

    /// Returns the rejection vote reason for the transaction at the specified consensus position. The result will be `None` when:
    /// * this node has never casted a reject vote for the transaction in question (either accepted or not processed it).
    /// * the transaction vote reason has been cleaned up due to the retention policy.
    pub fn get_rejection_vote_reason(&self, position: ConsensusPosition) -> Option<SomaError> {
        debug_assert_eq!(position.epoch, self.epoch, "Epoch mismatch");
        self.cache.read().get(&position).cloned()
    }

    /// Sets the last committed leader round. This is used to clean up the cache based on the retention policy.
    pub fn set_last_committed_leader_round(&self, round: u32) {
        let cut_off_round = round.saturating_sub(self.retention_rounds) + 1;
        let cut_off_position = ConsensusPosition {
            epoch: self.epoch,
            block: BlockRef::new(cut_off_round, AuthorityIndex::MIN, BlockDigest::MIN),
            index: TransactionIndex::MIN,
        };

        let mut cache = self.cache.write();
        let remaining = cache.split_off(&cut_off_position);
        trace!("Cleaned up {} entries", cache.len());
        *cache = remaining;
    }
}
