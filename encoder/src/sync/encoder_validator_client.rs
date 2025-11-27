use anyhow::{anyhow, Result};
use encoder_validator_api::tonic_gen::encoder_validator_api_client::EncoderValidatorApiClient;
use std::sync::Arc;
use tonic::transport::Channel;
use tracing::{debug, info, warn};
use types::{
    checkpoints::{CertifiedCheckpointSummary, EndOfEpochData},
    client::connect,
    committee::{Committee, EpochId, NetworkingCommittee},
    crypto::NetworkPublicKey,
    encoder_committee::EncoderCommittee,
    encoder_validator::{FetchCommitteesRequest, FetchCommitteesResponse, GetLatestEpochRequest},
    multiaddr::Multiaddr,
};

pub use super::utils::{
    extract_network_info_from_committees, extract_network_peers, NetworkPeerInfo,
};

/// Verified committees for an epoch, extracted from a certified checkpoint
#[derive(Clone, Debug)]
pub struct VerifiedEpochCommittees {
    pub epoch: EpochId,
    pub validator_committee: Committee,
    pub encoder_committee: EncoderCommittee,
    pub networking_committee: NetworkingCommittee,
    pub epoch_start_timestamp_ms: u64,
}

/// Client for fetching and verifying committees from validators
pub struct EncoderValidatorClient {
    client: EncoderValidatorApiClient<Channel>,

    /// The committee used to verify the next epoch's checkpoint
    current_validator_committee: Committee,

    /// Current epoch we've verified up to
    current_epoch: EpochId,

    /// Cached verified committees (current and optionally previous)
    verified_committees: Option<VerifiedEpochCommittees>,
    previous_committees: Option<VerifiedEpochCommittees>,
}

impl EncoderValidatorClient {
    /// Create a new client that connects to the validator node at the given address.
    ///
    /// # Arguments
    /// * `address` - The multiaddr of the validator node
    /// * `genesis_committee` - The genesis validator committee (root of trust)
    /// * `validator_network_key` - The network public key of the validator for TLS verification
    pub async fn new(
        address: &Multiaddr,
        genesis_committee: Committee,
        validator_network_key: NetworkPublicKey,
    ) -> Result<Self> {
        info!(
            "Creating encoder validator client connecting to {}",
            address
        );

        // Create TLS config targeting the validator's network key
        let tls_config = soma_tls::create_rustls_client_config(
            validator_network_key.into_inner(),
            soma_tls::SERVER_NAME.to_string(),
            None,
        );

        let channel = connect(address, tls_config)
            .await
            .map_err(|e| anyhow!("Failed to connect to validator: {}", e))?;

        let client = EncoderValidatorApiClient::new(channel);

        Ok(Self {
            client,
            current_validator_committee: genesis_committee,
            current_epoch: 0,
            verified_committees: None,
            previous_committees: None,
        })
    }

    /// Get the current epoch from the validator
    pub async fn get_current_epoch(&mut self) -> Result<EpochId> {
        let request = tonic::Request::new(GetLatestEpochRequest {
            start: self.current_epoch,
        });
        let response = self.client.get_latest_epoch(request).await?;
        Ok(response.get_ref().epoch)
    }

    /// Fetch certified checkpoints for epoch range
    async fn fetch_checkpoints(
        &mut self,
        start: EpochId,
        end: EpochId,
    ) -> Result<Vec<CertifiedCheckpointSummary>> {
        if start == 0 {
            return Err(anyhow!("Cannot fetch epoch 0 - genesis must be configured"));
        }

        info!("Fetching checkpoints for epochs {} to {}", start, end);
        let request = tonic::Request::new(FetchCommitteesRequest { start, end });
        let response = self.client.fetch_committees(request).await?;

        let mut checkpoints = response.into_inner().epoch_committees;
        checkpoints.sort_by_key(|c| c.data().epoch);

        Ok(checkpoints)
    }

    /// Verify a checkpoint and extract the next epoch's committees
    fn verify_and_extract_committees(
        &self,
        checkpoint: &CertifiedCheckpointSummary,
        verifying_committee: &Committee,
    ) -> Result<VerifiedEpochCommittees> {
        // Verify the checkpoint signature against the provided committee
        checkpoint
            .verify_authority_signatures(verifying_committee)
            .map_err(|e| anyhow!("Checkpoint signature verification failed: {}", e))?;

        // Extract end-of-epoch data
        let end_of_epoch = checkpoint
            .data()
            .end_of_epoch_data
            .as_ref()
            .ok_or_else(|| anyhow!("Checkpoint missing end_of_epoch_data"))?;

        // The checkpoint is for epoch N, and contains committees for epoch N+1
        let next_epoch = checkpoint.data().epoch + 1;

        Ok(VerifiedEpochCommittees {
            epoch: next_epoch,
            validator_committee: end_of_epoch.next_epoch_validator_committee.clone(),
            encoder_committee: end_of_epoch.next_epoch_encoder_committee.clone(),
            networking_committee: end_of_epoch.next_epoch_networking_committee.clone(),
            epoch_start_timestamp_ms: checkpoint.data().timestamp_ms,
        })
    }

    /// Sync committees from current epoch to target epoch
    pub async fn sync_to_epoch(
        &mut self,
        target_epoch: EpochId,
    ) -> Result<Option<VerifiedEpochCommittees>> {
        if self.current_epoch >= target_epoch {
            debug!(
                "Already at epoch {}, target is {}",
                self.current_epoch, target_epoch
            );
            return Ok(self.verified_committees.clone());
        }

        let start_epoch = self.current_epoch.max(1);
        let end_epoch = target_epoch - 1;

        if start_epoch > end_epoch {
            return Ok(None);
        }

        const CHUNK_SIZE: u64 = 10;
        let mut chunk_start = start_epoch;

        while chunk_start <= end_epoch {
            let chunk_end = (chunk_start + CHUNK_SIZE - 1).min(end_epoch);

            let checkpoints = self.fetch_checkpoints(chunk_start, chunk_end).await?;

            if checkpoints.is_empty() {
                warn!(
                    "No checkpoints returned for range {} to {}",
                    chunk_start, chunk_end
                );
                break;
            }

            for checkpoint in checkpoints {
                let checkpoint_epoch = checkpoint.data().epoch;

                if checkpoint_epoch < self.current_epoch {
                    continue;
                }

                let verified = self.verify_and_extract_committees(
                    &checkpoint,
                    &self.current_validator_committee,
                )?;

                info!(
                    "Verified checkpoint for epoch {}, extracted committees for epoch {}",
                    checkpoint_epoch, verified.epoch
                );

                self.previous_committees = self.verified_committees.take();
                self.verified_committees = Some(verified.clone());
                self.current_validator_committee = verified.validator_committee.clone();
                self.current_epoch = verified.epoch;
            }

            chunk_start = self.current_epoch;
        }

        Ok(self.verified_committees.clone())
    }

    /// Initial setup from genesis
    pub async fn setup_from_genesis(&mut self) -> Result<Option<VerifiedEpochCommittees>> {
        let target_epoch = self.get_current_epoch().await?;

        if target_epoch == 0 {
            info!("System is still in genesis epoch");
            return Ok(None);
        }

        self.sync_to_epoch(target_epoch).await
    }

    /// Poll for new committees
    pub async fn poll_for_updates(&mut self) -> Result<Option<VerifiedEpochCommittees>> {
        let latest_epoch = self.get_current_epoch().await?;

        if latest_epoch > self.current_epoch {
            self.sync_to_epoch(latest_epoch).await
        } else {
            Ok(None)
        }
    }

    // Accessors
    pub fn current_epoch(&self) -> EpochId {
        self.current_epoch
    }
    pub fn current_committees(&self) -> Option<&VerifiedEpochCommittees> {
        self.verified_committees.as_ref()
    }
    pub fn previous_committees(&self) -> Option<&VerifiedEpochCommittees> {
        self.previous_committees.as_ref()
    }
}
