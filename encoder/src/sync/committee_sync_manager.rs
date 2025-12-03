use crate::{
    sync::{
        encoder_validator_client::{
            EncoderValidatorClient, NetworkPeerInfo, VerifiedEpochCommittees,
        },
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
            if let Err(e) = manager_clone.run().await {
                error!("Committee sync loop failed: {:?}", e);
            }
        });

        manager
    }

    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    async fn run(&self) -> ShardResult<()> {
        // Initial sync
        self.initial_sync().await?;

        let mut backoff = Duration::from_millis(100);
        let max_backoff = Duration::from_secs(60);

        while !self.shutdown.load(Ordering::SeqCst) {
            let sleep_duration = self.calculate_sleep_duration();

            if sleep_duration > Duration::ZERO {
                tokio::time::sleep(sleep_duration).await;
                continue;
            }

            // Near epoch transition, poll frequently
            match self.try_sync().await {
                Ok(true) => {
                    backoff = Duration::from_millis(100);
                }
                Ok(false) => {
                    // No new epoch yet, wait a bit
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                Err(e) => {
                    error!("Sync error: {:?}", e);
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(max_backoff);
                }
            }
        }

        Ok(())
    }

    fn calculate_sleep_duration(&self) -> Duration {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let next_epoch_ms = self.next_epoch_time_ms.load(Ordering::SeqCst);

        if now_ms >= next_epoch_ms {
            return Duration::ZERO;
        }

        let time_until_epoch = next_epoch_ms - now_ms;
        let threshold = self.epoch_duration_ms / 10;

        if time_until_epoch > threshold {
            Duration::from_millis(time_until_epoch - threshold)
        } else {
            Duration::ZERO
        }
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

    async fn try_sync(&self) -> ShardResult<bool> {
        let mut client = self.validator_client.lock().await;

        match client.poll_for_updates().await {
            Ok(Some(committees)) => {
                drop(client);
                self.apply_committees(&committees)?;
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(e) => Err(ShardError::Other(format!("Poll failed: {}", e))),
        }
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
            self.context.inner().current_committees().vdf_iterations,
        );

        // Update allowed public keys
        self.update_allowed_keys(committees);

        // Update networking info
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
