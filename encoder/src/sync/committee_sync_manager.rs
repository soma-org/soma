use crate::{
    sync::encoder_validator_client::{EncoderValidatorClient, EnrichedVerifiedCommittees},
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
use types::shard_crypto::keys::{EncoderPublicKey, PeerPublicKey};
use types::shard_networking::EncoderNetworkingInfo;

/// Manager for committee synchronization with validator nodes
pub struct CommitteeSyncManager {
    /// Client for validator communication
    validator_client: Arc<Mutex<EncoderValidatorClient>>,

    /// Context to update with new committee information
    context: Context,

    /// Network information to update
    networking_info: EncoderNetworkingInfo,

    /// AllowPublicKeys to update with peer keys
    allower: AllowPublicKeys,

    /// Duration of each epoch in milliseconds
    epoch_duration_ms: u64,

    /// Current epoch being tracked
    current_epoch: AtomicU64,

    /// Timestamp when the next epoch is expected to start
    next_epoch_time_ms: AtomicU64,

    /// Own public key
    own_encoder_key: EncoderPublicKey,

    /// Signal for shutdown
    shutdown: Arc<AtomicBool>,
}

impl CommitteeSyncManager {
    /// Create a new committee sync manager
    pub fn new(
        validator_client: Arc<Mutex<EncoderValidatorClient>>,
        context: Context,
        networking_info: EncoderNetworkingInfo,
        allower: AllowPublicKeys,
        epoch_start_timestamp_ms: u64,
        epoch_duration_ms: u64,
        own_encoder_key: EncoderPublicKey,
    ) -> Self {
        let inner_context = context.inner();
        let current_epoch = inner_context.current_epoch;

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

    /// Start the committee synchronization manager
    pub async fn start(self) -> Arc<Self> {
        let manager = Arc::new(self);
        let manager_clone = manager.clone();

        // Spawn background task for committee synchronization
        tokio::spawn(async move {
            if let Err(e) = manager_clone.sync_loop().await {
                error!("Committee sync loop failed: {:?}", e);
            }
        });

        manager
    }

    /// Stop the synchronization manager
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Main synchronization loop
    async fn sync_loop(&self) -> ShardResult<()> {
        // Perform initial synchronization
        self.initial_sync().await?;

        // Continue polling for updates
        let mut backoff = Duration::from_millis(100);
        let max_backoff = Duration::from_secs(60);

        while !self.shutdown.load(Ordering::SeqCst) {
            // Get current time
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let next_epoch_ms = self.next_epoch_time_ms.load(Ordering::SeqCst);

            // If it's not yet time for the next epoch
            if now_ms < next_epoch_ms {
                // Calculate time until next epoch start
                let sleep_time = next_epoch_ms.saturating_sub(now_ms);

                // If we're more than 10% of an epoch away, sleep until we're closer
                if sleep_time > self.epoch_duration_ms / 10 {
                    // Sleep until we're within 10% of the epoch duration from the next epoch
                    let sleep_duration = sleep_time.saturating_sub(self.epoch_duration_ms / 10);
                    info!(
                        "Next epoch in {}ms, sleeping for {}ms until closer to transition",
                        sleep_time, sleep_duration
                    );
                    tokio::time::sleep(Duration::from_millis(sleep_duration)).await;
                    continue;
                }

                // Within 10% of next epoch - start polling more frequently
                debug!(
                    "Within 10% of epoch transition ({}ms away), polling more frequently",
                    sleep_time
                );
            }

            // Poll more frequently near epoch transition
            let poll_interval = std::cmp::max(self.epoch_duration_ms / 100, 1000); // At least 1 second

            // Poll for latest committees
            match self.poll_latest_committees().await {
                Ok(verified_committees) => {
                    let current_epoch = self.current_epoch.load(Ordering::SeqCst);

                    if verified_committees.validator_committee.epoch() > current_epoch {
                        // New epoch detected, update system state
                        if let Err(e) = self.apply_committees(verified_committees).await {
                            error!("Failed to apply committees: {:?}", e);

                            tokio::time::sleep(backoff).await;
                            backoff = std::cmp::min(backoff * 2, max_backoff);
                        } else {
                            // Reset backoff on success
                            backoff = Duration::from_millis(100);

                            // After a successful update, we can wait until the next epoch
                            continue;
                        }
                    }
                }
                Err(e) => {
                    // Apply exponential backoff on error
                    error!("Failed to poll latest committees: {:?}", e);

                    tokio::time::sleep(backoff).await;
                    backoff = std::cmp::min(backoff * 2, max_backoff);
                }
            }

            // Sleep with polling interval
            tokio::time::sleep(Duration::from_millis(poll_interval)).await;
        }

        Ok(())
    }

    /// Perform initial synchronization from genesis to current epoch

    async fn initial_sync(&self) -> ShardResult<()> {
        info!("Starting initial committee synchronization");

        // Get the current epoch from the validator
        let current_epoch = {
            let mut client = self.validator_client.lock().await;
            client
                .get_current_epoch()
                .await
                .map_err(|e| ShardError::Other(format!("Failed to get current epoch: {}", e)))?
        };

        // Check if we're still in genesis epoch
        if current_epoch == 0 {
            info!("System is still in genesis epoch (0) - skipping initial sync");

            info!(
                "Genesis epoch (0), next epoch expected around {:?}ms",
                self.next_epoch_time_ms
            );
            return Ok(());
        }

        // Normal path for non-genesis epochs
        let mut client = self.validator_client.lock().await;
        let verified_committees = client
            .setup_from_genesis()
            .await
            .map_err(|e| ShardError::Other(format!("Failed to setup from genesis: {}", e)))?;

        // Apply the committees
        drop(client); // Release lock before applying
        self.apply_committees(verified_committees).await?;

        info!(
            "Initial committee synchronization complete for epoch {}",
            current_epoch
        );
        Ok(())
    }

    /// Poll for latest committees
    async fn poll_latest_committees(&self) -> Result<EnrichedVerifiedCommittees, ShardError> {
        let mut client = self.validator_client.lock().await;
        client
            .poll_latest_committees()
            .await
            .map_err(|e| ShardError::Other(format!("Failed to poll latest committees: {}", e)))
    }

    /// Apply verified committees to system state
    async fn apply_committees(&self, committees: EnrichedVerifiedCommittees) -> ShardResult<()> {
        let epoch = committees.validator_committee.epoch();

        // Create new Committees struct for the upcoming epoch
        let new_committees = Committees::new(
            epoch,
            committees.validator_committee.clone(),
            committees.encoder_committee.clone(),
            committees.networking_committee.clone(),
            self.context.inner().current_committees().vdf_iterations, // Keep the same VDF iterations
        );

        // Update AllowPublicKeys with all peer keys from both current and previous epochs
        self.update_allowed_public_keys(&committees);

        // Update EncoderNetworkingInfo with the pre-extracted data
        if !committees.networking_info.is_empty() {
            info!(
                "Updating networking info with {} entries",
                committees.networking_info.len()
            );
            self.networking_info.update(committees.networking_info);
        }

        // Get current context inner
        let inner_context = self.context.inner();

        // Create new InnerContext with updated committees array
        // Simply use the complete object_servers from the enriched committees
        let new_inner_context = InnerContext::new(
            [
                inner_context
                    .committees(inner_context.current_epoch)?
                    .clone(),
                new_committees,
            ],
            epoch, // New current epoch
            self.own_encoder_key.clone(),
        );

        // Update the context with the new inner context
        self.context.update(new_inner_context);

        // Update tracked epoch
        self.current_epoch.store(epoch, Ordering::SeqCst);

        // Calculate the timestamp when the next epoch will start
        // This is current epoch's start time + epoch duration
        let next_epoch_time_ms = committees.epoch_start_timestamp_ms + self.epoch_duration_ms;
        self.next_epoch_time_ms
            .store(next_epoch_time_ms, Ordering::SeqCst);

        info!(
            "Updated committee for epoch {}, epoch started at {}ms, next epoch expected at {}ms",
            epoch, committees.epoch_start_timestamp_ms, next_epoch_time_ms
        );

        Ok(())
    }

    // Update AllowPublicKeys with all peer keys from the committees
    fn update_allowed_public_keys(&self, committees: &EnrichedVerifiedCommittees) {
        // Extract all peer public keys from connections_info
        let mut allowed_keys = BTreeSet::new();

        for (_, (peer_key, _)) in committees.networking_info.clone() {
            // Add inner key to allowed set
            allowed_keys.insert(peer_key.clone().into_inner());
        }

        // Extract network keys from the NetworkingCommittee
        for (_, network_metadata) in committees.networking_committee.members() {
            info!(
                "Updating allowed keys with networking committee member {}",
                network_metadata.hostname
            );
            allowed_keys.insert(network_metadata.network_key.clone().into_inner());
        }

        info!(
            "Updating allowed public keys with {} entries",
            allowed_keys.len()
        );

        // Update the allower with the complete set of keys
        self.allower.update(allowed_keys);
    }
}
