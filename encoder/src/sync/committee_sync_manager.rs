use crate::{
    sync::{
        encoder_validator_client::{EncoderValidatorClient, VerifiedEpochCommittees},
        utils::extract_network_peers,
    },
    types::context::{Committees, Context, InnerContext},
};
use soma_tls::AllowPublicKeys;
use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};
use types::error::{ShardError, ShardResult};
use types::shard_crypto::keys::EncoderPublicKey;
use types::shard_networking::EncoderNetworkingInfo;

pub struct CommitteeSyncManager {
    validator_client: Arc<Mutex<EncoderValidatorClient>>,
    context: Context,
    networking_info: EncoderNetworkingInfo,
    allower: AllowPublicKeys,
    epoch_duration_ms: u64,
    current_epoch: AtomicU64,
    next_epoch_time_ms: AtomicU64,
    own_encoder_key: EncoderPublicKey,
    shutdown: Arc<AtomicBool>,
}

impl CommitteeSyncManager {
    pub fn new(
        validator_client: Arc<Mutex<EncoderValidatorClient>>,
        context: Context,
        networking_info: EncoderNetworkingInfo,
        allower: AllowPublicKeys,
        epoch_start_timestamp_ms: u64,
        epoch_duration_ms: u64,
        own_encoder_key: EncoderPublicKey,
    ) -> Self {
        let current_epoch = context.inner().current_epoch;

        Self {
            validator_client,
            context,
            networking_info,
            allower,
            epoch_duration_ms,
            current_epoch: AtomicU64::new(current_epoch),
            next_epoch_time_ms: AtomicU64::new(epoch_start_timestamp_ms + epoch_duration_ms),
            own_encoder_key,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn start(self) -> Arc<Self> {
        let manager = Arc::new(self);
        let manager_clone = manager.clone();

        tokio::spawn(async move {
            if let Err(e) = manager_clone.sync_loop().await {
                error!("Committee sync loop failed: {:?}", e);
            }
        });

        manager
    }

    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    async fn sync_loop(&self) -> ShardResult<()> {
        // Initial sync
        self.initial_sync().await?;

        let mut backoff = Duration::from_millis(100);
        let max_backoff = Duration::from_secs(60);

        while !self.shutdown.load(Ordering::SeqCst) {
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let next_epoch_ms = self.next_epoch_time_ms.load(Ordering::SeqCst);

            // If it's not yet time for the next epoch
            if now_ms < next_epoch_ms {
                let sleep_time = next_epoch_ms.saturating_sub(now_ms);

                // If we're more than 10% of an epoch away, sleep until we're closer
                if sleep_time > self.epoch_duration_ms / 10 {
                    let sleep_duration = sleep_time.saturating_sub(self.epoch_duration_ms / 10);
                    info!(
                        "Next epoch in {}ms, sleeping for {}ms until closer to transition",
                        sleep_time, sleep_duration
                    );
                    tokio::time::sleep(Duration::from_millis(sleep_duration)).await;
                    continue;
                }

                debug!(
                    "Within 10% of epoch transition ({}ms away), polling more frequently",
                    sleep_time
                );
            }

            // Poll interval: at least 1 second, or 1% of epoch duration
            let poll_interval = std::cmp::max(self.epoch_duration_ms / 100, 1000);

            // Poll for updates
            match self.poll_for_updates().await {
                Ok(Some(committees)) => {
                    let current_epoch = self.current_epoch.load(Ordering::SeqCst);

                    // Only apply if this is actually a newer epoch
                    if committees.epoch > current_epoch {
                        if let Err(e) = self.apply_committees(&committees) {
                            error!("Failed to apply committees: {:?}", e);
                            tokio::time::sleep(backoff).await;
                            backoff = std::cmp::min(backoff * 2, max_backoff);
                        } else {
                            // Reset backoff on success
                            backoff = Duration::from_millis(100);
                            continue;
                        }
                    }
                }
                Ok(None) => {
                    // No new epoch yet, continue polling
                }
                Err(e) => {
                    error!("Failed to poll for updates: {:?}", e);
                    tokio::time::sleep(backoff).await;
                    backoff = std::cmp::min(backoff * 2, max_backoff);
                }
            }

            tokio::time::sleep(Duration::from_millis(poll_interval)).await;
        }

        Ok(())
    }

    async fn initial_sync(&self) -> ShardResult<()> {
        info!("Starting initial committee synchronization");

        let mut client = self.validator_client.lock().await;

        match client.setup_from_genesis().await {
            Ok(Some(committees)) => {
                drop(client);
                self.apply_committees(&committees)?;
                info!("Initial sync complete for epoch {}", committees.epoch);
            }
            Ok(None) => {
                info!("Genesis epoch - no sync needed");
            }
            Err(e) => {
                return Err(ShardError::Other(format!("Initial sync failed: {}", e)));
            }
        }

        Ok(())
    }

    async fn poll_for_updates(&self) -> ShardResult<Option<VerifiedEpochCommittees>> {
        let mut client = self.validator_client.lock().await;
        client
            .poll_for_updates()
            .await
            .map_err(|e| ShardError::Other(format!("Poll failed: {}", e)))
    }

    fn apply_committees(&self, committees: &VerifiedEpochCommittees) -> ShardResult<()> {
        let epoch = committees.epoch;
        info!("Applying committees for epoch {}", epoch);

        // Create new Committees struct
        let new_committees = Committees::new(
            epoch,
            committees.validator_committee.clone(),
            committees.encoder_committee.clone(),
            committees.networking_committee.clone(),
            committees.vdf_iterations,
        );

        // Update allowed public keys
        self.update_allowed_keys(committees);

        // Extract and update networking info from encoder committee
        let peers = extract_network_peers(&committees.encoder_committee);
        let networking_entries: Vec<_> = peers
            .iter()
            .map(|p| {
                (
                    p.encoder_key.clone(),
                    (p.network_key.clone(), p.internal_address.clone()),
                )
            })
            .collect();

        if !networking_entries.is_empty() {
            info!(
                "Updating networking info with {} entries",
                networking_entries.len()
            );
            self.networking_info.update(networking_entries);
        }

        // Update context
        let inner = self.context.inner();
        let new_inner = InnerContext::new(
            [
                inner.committees(inner.current_epoch)?.clone(),
                new_committees,
            ],
            epoch,
            self.own_encoder_key.clone(),
            self.context.own_network_keypair(),
        );
        self.context.update(new_inner);

        // Update tracking state
        self.current_epoch.store(epoch, Ordering::SeqCst);
        let next_epoch_time = committees.epoch_start_timestamp_ms + self.epoch_duration_ms;
        self.next_epoch_time_ms
            .store(next_epoch_time, Ordering::SeqCst);

        info!(
            "Applied epoch {}, next epoch expected at {}ms",
            epoch, next_epoch_time
        );

        Ok(())
    }

    fn update_allowed_keys(&self, committees: &VerifiedEpochCommittees) {
        let mut allowed = BTreeSet::new();

        // From encoder committee
        let peers = extract_network_peers(&committees.encoder_committee);
        for peer in &peers {
            allowed.insert(peer.network_key.clone().into_inner());
        }

        // From networking committee
        for (_, metadata) in committees.networking_committee.members() {
            allowed.insert(metadata.network_key.clone().into_inner());
        }

        info!("Updating allowed keys with {} entries", allowed.len());
        self.allower.update(allowed);
    }

    pub fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::SeqCst)
    }
}
