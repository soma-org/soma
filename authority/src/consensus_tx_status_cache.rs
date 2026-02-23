// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use parking_lot::{RwLock, RwLockWriteGuard};
use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};
use tokio::sync::watch;
use tracing::debug;
use types::{
    consensus::{ConsensusPosition, block::Round},
    error::{SomaError, SomaResult},
};
use utils::notify_read::NotifyRead;

/// The number of consensus rounds to retain transaction status information before garbage collection.
/// Used to expire positions from old rounds, as well as to check if a transaction is too far ahead of the last committed round.
/// Assuming a max round rate of 15/sec, this allows status updates to be valid within a window of ~25-30 seconds.
pub(crate) const CONSENSUS_STATUS_RETENTION_ROUNDS: u32 = 400;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConsensusTxStatus {
    // Transaction is voted to accept by a quorum of validators on fastpath.
    FastpathCertified,
    // Transaction is rejected, either by a quorum of validators or indirectly post-commit.
    Rejected,
    // Transaction is finalized post commit.
    Finalized,
}

#[derive(Debug, Clone)]
pub(crate) enum NotifyReadConsensusTxStatusResult {
    // The consensus position to be read has been updated with a new status.
    Status(ConsensusTxStatus),
    // The consensus position to be read has expired.
    // Provided with the last committed round that was used to check for expiration.
    Expired(u32),
}

pub(crate) struct ConsensusTxStatusCache {
    // GC depth in consensus.
    consensus_gc_depth: u32,

    inner: RwLock<Inner>,

    status_notify_read: NotifyRead<ConsensusPosition, ConsensusTxStatus>,
    /// Watch channel for last committed leader round updates
    last_committed_leader_round_tx: watch::Sender<Option<u32>>,
    last_committed_leader_round_rx: watch::Receiver<Option<u32>>,
}

#[derive(Default)]
struct Inner {
    /// A map of transaction position to its status from consensus.
    transaction_status: BTreeMap<ConsensusPosition, ConsensusTxStatus>,
    /// Consensus positions that are currently in the fastpath certified state.
    fastpath_certified: BTreeSet<ConsensusPosition>,
    /// The last leader round updated in update_last_committed_leader_round().
    last_committed_leader_round: Option<Round>,
}

impl ConsensusTxStatusCache {
    pub(crate) fn new(consensus_gc_depth: Round) -> Self {
        assert!(
            consensus_gc_depth < CONSENSUS_STATUS_RETENTION_ROUNDS,
            "{} vs {}",
            consensus_gc_depth,
            CONSENSUS_STATUS_RETENTION_ROUNDS
        );
        let (last_committed_leader_round_tx, last_committed_leader_round_rx) = watch::channel(None);
        Self {
            consensus_gc_depth,
            inner: Default::default(),
            status_notify_read: Default::default(),
            last_committed_leader_round_tx,
            last_committed_leader_round_rx,
        }
    }

    pub(crate) fn set_transaction_status(&self, pos: ConsensusPosition, status: ConsensusTxStatus) {
        if let Some(last_committed_leader_round) = *self.last_committed_leader_round_rx.borrow() {
            if pos.block.round + CONSENSUS_STATUS_RETENTION_ROUNDS <= last_committed_leader_round {
                // Ignore stale status updates.
                return;
            }
        }

        let mut inner = self.inner.write();
        self.set_transaction_status_inner(&mut inner, pos, status);
    }

    fn set_transaction_status_inner(
        &self,
        inner: &mut RwLockWriteGuard<Inner>,
        pos: ConsensusPosition,
        status: ConsensusTxStatus,
    ) {
        // Calls to set_transaction_status are async and can be out of order.
        // Makes sure this is tolerated by handling state transitions properly.
        let status_entry = inner.transaction_status.entry(pos);
        match status_entry {
            Entry::Vacant(entry) => {
                // Set the status for the first time.
                entry.insert(status);
                if status == ConsensusTxStatus::FastpathCertified {
                    // Only path where a status can be set to fastpath certified.
                    assert!(inner.fastpath_certified.insert(pos));
                }
            }
            Entry::Occupied(mut entry) => {
                let old_status = *entry.get();
                match (old_status, status) {
                    // If the statuses are the same, no update is needed.
                    (s1, s2) if s1 == s2 => return,
                    // FastpathCertified is transient and can be updated to other statuses.
                    (ConsensusTxStatus::FastpathCertified, _) => {
                        entry.insert(status);
                        if old_status == ConsensusTxStatus::FastpathCertified {
                            // Only path where a status can transition out of fastpath certified.
                            assert!(inner.fastpath_certified.remove(&pos));
                        }
                    }
                    // This happens when statuses arrive out-of-order, and is a no-op.
                    (
                        ConsensusTxStatus::Rejected | ConsensusTxStatus::Finalized,
                        ConsensusTxStatus::FastpathCertified,
                    ) => {
                        return;
                    }
                    // Transitions between terminal statuses are invalid.
                    _ => {
                        panic!(
                            "Conflicting status updates for transaction {:?}: {:?} -> {:?}",
                            pos, old_status, status
                        );
                    }
                }
            }
        };

        // All code paths leading to here should have set the status.
        debug!("Transaction status is set for {}: {:?}", pos, status);
        self.status_notify_read.notify(&pos, &status);
    }

    /// Given a known previous status provided by `old_status`, this function will return a new
    /// status once the transaction status has changed, or if the consensus position has expired.
    pub(crate) async fn notify_read_transaction_status_change(
        &self,
        consensus_position: ConsensusPosition,
        old_status: Option<ConsensusTxStatus>,
    ) -> NotifyReadConsensusTxStatusResult {
        // TODO(fastpath): We should track the typical distance between the last committed round
        // and the requested round notified as metrics.
        let registration = self.status_notify_read.register_one(&consensus_position);
        let mut round_rx = self.last_committed_leader_round_rx.clone();
        {
            let inner = self.inner.read();
            if let Some(status) = inner.transaction_status.get(&consensus_position) {
                if Some(status) != old_status.as_ref() {
                    if let Some(old_status) = old_status {
                        // The only scenario where the status may change, is when the transaction
                        // is initially fastpath certified, and then later finalized or rejected.
                        assert_eq!(old_status, ConsensusTxStatus::FastpathCertified);
                    }
                    return NotifyReadConsensusTxStatusResult::Status(*status);
                }
            }
            // Inner read lock dropped here.
        }

        let expiration_check = async {
            loop {
                if let Some(last_committed_leader_round) = *round_rx.borrow() {
                    if consensus_position.block.round + CONSENSUS_STATUS_RETENTION_ROUNDS
                        <= last_committed_leader_round
                    {
                        return last_committed_leader_round;
                    }
                }
                // Channel closed - this should never happen in practice, so panic
                round_rx
                    .changed()
                    .await
                    .expect("last_committed_leader_round watch channel closed unexpectedly");
            }
        };
        tokio::select! {
            status = registration => NotifyReadConsensusTxStatusResult::Status(status),
            last_committed_leader_round = expiration_check => NotifyReadConsensusTxStatusResult::Expired(last_committed_leader_round),
        }
    }

    pub(crate) async fn update_last_committed_leader_round(
        &self,
        last_committed_leader_round: u32,
    ) {
        debug!("Updating last committed leader round: {}", last_committed_leader_round);

        let mut inner = self.inner.write();

        // Consensus only bumps GC round after generating a commit. So if we expire and GC transactions
        // based on the latest committed leader round, we may expire transactions in the current commit, or
        // make these transactions' statuses very short lived.
        // So we only expire and GC transactions with the previous committed leader round.
        let Some(leader_round) =
            inner.last_committed_leader_round.replace(last_committed_leader_round)
        else {
            // This is the first update. Do not expire or GC any transactions.
            return;
        };

        // Remove transactions that are expired.
        while let Some((position, _)) = inner.transaction_status.first_key_value() {
            if position.block.round + CONSENSUS_STATUS_RETENTION_ROUNDS <= leader_round {
                let (pos, status) = inner.transaction_status.pop_first().unwrap();
                // Ensure the transaction is not in the fastpath certified set.
                if status == ConsensusTxStatus::FastpathCertified {
                    assert!(inner.fastpath_certified.remove(&pos));
                }
            } else {
                break;
            }
        }

        // GC fastpath certified transactions.
        // In theory, notify_read_transaction_status_change() could return `Rejected` status directly
        // to waiters on GC'ed transactions.
        // But it is necessary to track the number of fastpath certified status anyway for end of epoch.
        // So rejecting every fastpath certified transaction here.
        while let Some(position) = inner.fastpath_certified.first().cloned() {
            if position.block.round + self.consensus_gc_depth <= leader_round {
                // Reject GC'ed transactions that were previously fastpath certified.
                self.set_transaction_status_inner(
                    &mut inner,
                    position,
                    ConsensusTxStatus::Rejected,
                );
            } else {
                break;
            }
        }

        // Send update through watch channel.
        let _ = self.last_committed_leader_round_tx.send(Some(leader_round));
    }

    pub(crate) fn get_last_committed_leader_round(&self) -> Option<u32> {
        *self.last_committed_leader_round_rx.borrow()
    }

    pub(crate) fn get_num_fastpath_certified(&self) -> usize {
        self.inner.read().fastpath_certified.len()
    }

    /// Returns true if the position is too far ahead of the last committed round.
    pub(crate) fn check_position_too_ahead(&self, position: &ConsensusPosition) -> SomaResult<()> {
        if let Some(last_committed_leader_round) = *self.last_committed_leader_round_rx.borrow() {
            if position.block.round
                > last_committed_leader_round + CONSENSUS_STATUS_RETENTION_ROUNDS
            {
                return Err(SomaError::ValidatorConsensusLagging {
                    round: position.block.round,
                    last_committed_round: last_committed_leader_round,
                }
                .into());
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn get_transaction_status(
        &self,
        position: &ConsensusPosition,
    ) -> Option<ConsensusTxStatus> {
        let inner = self.inner.read();
        inner.transaction_status.get(position).cloned()
    }
}
