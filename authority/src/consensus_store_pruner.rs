use std::fs;
use std::sync::Arc;
use std::{path::PathBuf, time::Duration};
use store::rocks::safe_drop_db;
use tokio::{sync::mpsc, time::Instant};
use tracing::{error, info, warn};
use types::committee::Epoch;
use types::storage::consensus::ConsensusStore;

pub struct ConsensusStorePruner {
    tx_prune: mpsc::Sender<Epoch>,
    _handle: tokio::task::JoinHandle<()>,
}

impl ConsensusStorePruner {
    pub fn new(
        consensus_store: Arc<dyn ConsensusStore>,
        epoch_retention: u64,
        epoch_prune_period: Duration,
    ) -> Self {
        let (tx_prune, mut rx_prune) = mpsc::channel(1);

        let _handle = tokio::task::spawn(async move {
            info!("Starting consensus store pruner with epoch retention {epoch_retention} and prune period {epoch_prune_period:?}");

            let mut timeout = tokio::time::interval_at(
                Instant::now() + Duration::from_secs(60), // allow some time for the node to boot
                epoch_prune_period,
            );

            let mut latest_epoch = 0;
            loop {
                tokio::select! {
                    _ = timeout.tick() => {
                        if latest_epoch > 0 {
                            Self::prune_old_epoch_data(&consensus_store, latest_epoch, epoch_retention).await;
                        }
                    }
                    result = rx_prune.recv() => {
                        if result.is_none() {
                            info!("Closing consensus store pruner");
                            break;
                        }
                        latest_epoch = result.unwrap();
                        Self::prune_old_epoch_data(&consensus_store, latest_epoch, epoch_retention).await;
                    }
                }
            }
        });

        Self { tx_prune, _handle }
    }

    /// This method will remove all epoch data that is older than the current epoch minus the epoch retention.
    pub async fn prune(&self, current_epoch: Epoch) {
        let result = self.tx_prune.send(current_epoch).await;
        if result.is_err() {
            error!(
                "Error sending message to data removal task for epoch {:?}",
                current_epoch,
            );
        }
    }

    async fn prune_old_epoch_data(
        consensus_store: &Arc<dyn ConsensusStore>,
        current_epoch: Epoch,
        epoch_retention: u64,
    ) {
        let drop_boundary = current_epoch.saturating_sub(epoch_retention);

        info!(
            "Consensus store pruning for current epoch {}. Will remove epochs < {:?}",
            current_epoch, drop_boundary
        );

        // Call the new prune_epochs method on the consensus store
        if let Err(e) = consensus_store.prune_epochs_before(drop_boundary) {
            error!("Failed to prune old epochs from consensus store: {:?}", e);
        } else {
            info!(
                "Successfully pruned epochs < {} from consensus store",
                drop_boundary
            );
        }
    }
}
