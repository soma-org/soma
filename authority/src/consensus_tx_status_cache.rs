// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use parking_lot::RwLock;
use tokio::sync::watch;
use tracing::debug;
use types::consensus::ConsensusPosition;
use types::consensus::block::Round;
use types::error::{SomaError, SomaResult};
use utils::notify_read::NotifyRead;

/// The number of consensus rounds to retain transaction status information before garbage collection.
/// Used to expire positions from old rounds, as well as to check if a transaction is too far ahead of the last committed round.
/// Assuming a max round rate of 15/sec, this allows status updates to be valid within a window of ~25-30 seconds.
pub(crate) const CONSENSUS_STATUS_RETENTION_ROUNDS: u32 = 400;

/// Terminal consensus statuses. Stage 5b removed the transient
/// `FastpathCertified` variant — every status now is final.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConsensusTxStatus {
    /// Transaction is rejected, either by a quorum of validators or indirectly post-commit.
    Rejected,
    /// Transaction is finalized post commit.
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
    /// The last leader round updated in update_last_committed_leader_round().
    last_committed_leader_round: Option<Round>,
}

impl ConsensusTxStatusCache {
    pub(crate) fn new(consensus_gc_depth: Round) -> Self {
        // `consensus_gc_depth` was previously used to GC fastpath-certified
        // entries. Stage 5b dropped FastpathCertified, so this argument is
        // no longer load-bearing — kept as a no-op for callsite stability.
        assert!(
            consensus_gc_depth < CONSENSUS_STATUS_RETENTION_ROUNDS,
            "{} vs {}",
            consensus_gc_depth,
            CONSENSUS_STATUS_RETENTION_ROUNDS
        );
        let (last_committed_leader_round_tx, last_committed_leader_round_rx) = watch::channel(None);
        Self {
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
        // Stage 5b: only terminal statuses (Rejected/Finalized) exist now.
        // Calls can still arrive out of order, but every status is final,
        // so the rule is simple: first writer wins, identical reposts are
        // no-ops, conflicting terminals are a bug.
        match inner.transaction_status.entry(pos) {
            Entry::Vacant(entry) => {
                entry.insert(status);
            }
            Entry::Occupied(entry) => {
                let old_status = *entry.get();
                if old_status == status {
                    // Identical re-post — no-op.
                    return;
                }
                panic!(
                    "Conflicting status updates for transaction {:?}: {:?} -> {:?}",
                    pos, old_status, status
                );
            }
        }

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
        // TODO: We should track the typical distance between the last committed round
        // and the requested round notified as metrics.
        let registration = self.status_notify_read.register_one(&consensus_position);
        let mut round_rx = self.last_committed_leader_round_rx.clone();
        {
            let inner = self.inner.read();
            if let Some(status) = inner.transaction_status.get(&consensus_position) {
                // Stage 5b: only terminal statuses exist now. If a caller
                // already has the same status, they're synchronizing to a
                // settled value — return immediately.
                if Some(status) != old_status.as_ref() {
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

        // Remove transactions that are expired. Stage 5b: no separate
        // fastpath-certified tracking — every entry is a terminal
        // (Rejected/Finalized) status, GC just drops it.
        while let Some((position, _)) = inner.transaction_status.first_key_value() {
            if position.block.round + CONSENSUS_STATUS_RETENTION_ROUNDS <= leader_round {
                inner.transaction_status.pop_first();
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
