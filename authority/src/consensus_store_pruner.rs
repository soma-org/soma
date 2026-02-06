use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use store::rocks::safe_drop_db;
use tokio::{sync::mpsc, time::Instant};
use tracing::{error, info, warn};
use types::committee::Epoch;

pub struct ConsensusStorePruner {
    tx_remove: mpsc::Sender<Epoch>,
    _handle: tokio::task::JoinHandle<()>,
}

impl ConsensusStorePruner {
    pub fn new(base_path: PathBuf, epoch_retention: u64, epoch_prune_period: Duration) -> Self {
        let (tx_remove, mut rx_remove) = mpsc::channel(1);

        let _handle = tokio::spawn(async move {
            info!(
                "Starting consensus store pruner with epoch retention {epoch_retention} and prune period {epoch_prune_period:?}"
            );

            let mut timeout = tokio::time::interval_at(
                Instant::now() + Duration::from_secs(60), // allow some time for the node to boot etc before attempting to prune
                epoch_prune_period,
            );

            let mut latest_epoch = 0;
            loop {
                tokio::select! {
                    _ = timeout.tick() => {
                        Self::prune_old_epoch_data(&base_path, latest_epoch, epoch_retention).await;
                    }
                    result = rx_remove.recv() => {
                        if result.is_none() {
                            info!("Closing consensus store pruner");
                            break;
                        }
                        latest_epoch = result.unwrap();
                        Self::prune_old_epoch_data(&base_path, latest_epoch, epoch_retention).await;
                    }
                }
            }
        });

        Self { tx_remove, _handle }
    }

    /// This method will remove all epoch data stores and directories that are older than the current epoch minus the epoch retention. The method ensures
    /// that always the `current_epoch` data is retained.
    pub async fn prune(&self, current_epoch: Epoch) {
        let result = self.tx_remove.send(current_epoch).await;
        if result.is_err() {
            error!("Error sending message to data removal task for epoch {:?}", current_epoch,);
        }
    }

    async fn prune_old_epoch_data(
        storage_base_path: &PathBuf,
        current_epoch: Epoch,
        epoch_retention: u64,
    ) {
        let drop_boundary = current_epoch.saturating_sub(epoch_retention);

        info!(
            "Consensus store prunning for current epoch {}. Will remove epochs < {:?}",
            current_epoch, drop_boundary
        );

        // Get all the epoch stores in the base path directory
        let files = match fs::read_dir(storage_base_path) {
            Ok(f) => f,
            Err(e) => {
                error!(
                    "Can not read the files in the storage path directory for epoch cleanup: {:?}",
                    e
                );
                return;
            }
        };

        // Look for any that are less than the drop boundary and drop
        for file_res in files {
            let f = match file_res {
                Ok(f) => f,
                Err(e) => {
                    error!("Error while cleaning up storage of previous epochs: {:?}", e);
                    continue;
                }
            };

            let name = f.file_name();
            let file_epoch_string = match name.to_str() {
                Some(f) => f,
                None => continue,
            };

            let file_epoch = match file_epoch_string.to_owned().parse::<u64>() {
                Ok(f) => f,
                Err(e) => {
                    error!(
                        "Could not parse file \"{file_epoch_string}\" in storage path into epoch for cleanup: {:?}",
                        e
                    );
                    continue;
                }
            };

            if file_epoch < drop_boundary {
                const WAIT_BEFORE_FORCE_DELETE: Duration = Duration::from_secs(5);
                if let Err(e) = safe_drop_db(f.path(), WAIT_BEFORE_FORCE_DELETE).await {
                    warn!(
                        "Could not prune old consensus storage \"{:?}\" directory with safe approach. Will fallback to force delete: {:?}",
                        f.path(),
                        e
                    );

                    if let Err(err) = fs::remove_dir_all(f.path()) {
                        error!(
                            "Could not prune old consensus storage \"{:?}\" directory with force delete: {:?}",
                            f.path(),
                            err
                        );
                    } else {
                        info!(
                            "Successfully pruned consensus epoch storage directory with force delete: {:?}",
                            f.path()
                        );
                    }
                } else {
                    info!("Successfully pruned consensus epoch storage directory: {:?}", f.path());
                }
            }
        }

        info!("Completed old epoch data removal process for epoch {:?}", current_epoch);
    }
}
