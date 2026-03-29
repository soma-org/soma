//! Soma checkpoint watcher.
//!
//! Watches Soma checkpoints for PendingWithdrawal objects created by
//! BridgeWithdraw transactions, and epoch boundary events for committee rotation.

use tokio::sync::mpsc;
use tracing::{info, warn};

use types::object::ObjectID;
use types::bridge::PendingWithdrawal;
use types::object::{Object, ObjectType};

use crate::error::{BridgeError, BridgeResult};
use crate::types::ObservedWithdrawal;

/// Events produced by the checkpoint watcher.
#[derive(Debug)]
pub enum CheckpointEvent {
    /// A new PendingWithdrawal was created on-chain.
    NewWithdrawal(ObservedWithdrawal),
    /// An epoch boundary was detected — the validator set (and thus bridge committee)
    /// may have changed.
    EpochBoundary { epoch: u64 },
}

/// Watches Soma checkpoints for bridge-relevant events.
///
/// This integrates with the existing `data-ingestion` framework:
/// - Subscribes to checkpoint data via the fullnode's checkpoint stream
/// - Scans each checkpoint's output objects for PendingWithdrawal
/// - Detects epoch boundaries for committee rotation
pub struct CheckpointWatcher {
    event_tx: mpsc::Sender<CheckpointEvent>,
    last_epoch: u64,
}

impl CheckpointWatcher {
    /// Create a new checkpoint watcher.
    /// Returns the watcher and a receiver for bridge events.
    pub fn new(buffer_size: usize) -> (Self, mpsc::Receiver<CheckpointEvent>) {
        let (tx, rx) = mpsc::channel(buffer_size);
        (
            Self {
                event_tx: tx,
                last_epoch: 0,
            },
            rx,
        )
    }

    /// Process a single checkpoint's worth of data.
    ///
    /// Called by the data ingestion framework for each checkpoint.
    /// Scans output objects for PendingWithdrawal and detects epoch boundaries.
    pub async fn process_checkpoint(
        &mut self,
        checkpoint_epoch: u64,
        created_objects: &[(ObjectID, Object)],
    ) -> BridgeResult<()> {
        // Detect epoch boundary
        if checkpoint_epoch > self.last_epoch && self.last_epoch > 0 {
            info!(
                old_epoch = self.last_epoch,
                new_epoch = checkpoint_epoch,
                "Epoch boundary detected"
            );
            self.event_tx
                .send(CheckpointEvent::EpochBoundary {
                    epoch: checkpoint_epoch,
                })
                .await
                .map_err(|_| BridgeError::Internal("Event channel closed".into()))?;
        }
        self.last_epoch = checkpoint_epoch;

        // Scan for PendingWithdrawal objects
        for (id, obj) in created_objects {
            if let Some(pw) = obj.deserialize_contents::<PendingWithdrawal>(
                ObjectType::PendingWithdrawal,
            ) {
                let withdrawal = ObservedWithdrawal {
                    id: *id,
                    nonce: pw.nonce,
                    sender: pw.sender,
                    recipient_eth_address: pw.recipient_eth_address,
                    amount: pw.amount,
                };
                info!(
                    nonce = pw.nonce,
                    amount = pw.amount,
                    "Observed PendingWithdrawal"
                );
                self.event_tx
                    .send(CheckpointEvent::NewWithdrawal(withdrawal))
                    .await
                    .map_err(|_| {
                        BridgeError::Internal("Event channel closed".into())
                    })?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_epoch_boundary_detection() {
        let (mut watcher, mut rx) = CheckpointWatcher::new(16);

        // First checkpoint at epoch 1 — no event (last_epoch is 0, initial)
        watcher
            .process_checkpoint(1, &[])
            .await
            .unwrap();

        // Second checkpoint still epoch 1 — no event
        watcher
            .process_checkpoint(1, &[])
            .await
            .unwrap();

        // Third checkpoint at epoch 2 — epoch boundary detected
        watcher
            .process_checkpoint(2, &[])
            .await
            .unwrap();

        match rx.try_recv() {
            Ok(CheckpointEvent::EpochBoundary { epoch }) => assert_eq!(epoch, 2),
            other => panic!("Expected EpochBoundary, got {:?}", other),
        }
    }
}
