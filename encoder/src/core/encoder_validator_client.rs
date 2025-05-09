// src/encoder_validator_client.rs

use crate::types::encoder_committee::{Encoder, EncoderCommittee as ShardCommittee};
use anyhow::{anyhow, Result};
use encoder_validator_api::tonic_gen::encoder_validator_api_client::EncoderValidatorApiClient;
use shared::{crypto::keys::EncoderPublicKey, probe::ProbeMetadata};
use std::{collections::BTreeMap, sync::Arc};
use tonic::transport::{Channel, Endpoint};
use tracing::{debug, info, warn};
use types::{
    base::AuthorityName,
    client::connect,
    committee::{
        to_encoder_committee_intent, Authority, AuthorityIndex, Committee, EncoderCommittee,
        EpochId,
    },
    consensus::{
        stake_aggregator::{QuorumThreshold, StakeAggregator},
        validator_set::{to_validator_set_intent, ValidatorSet},
    },
    crypto::{AggregateAuthenticator, AggregateAuthoritySignature, AuthorityPublicKey},
    encoder_validator::{
        EpochCommittee, FetchCommitteesRequest, FetchCommitteesResponse, GetLatestEpochRequest,
    },
    multiaddr::Multiaddr,
};

/// Result type for committee verification
pub struct VerifiedCommittees {
    pub validator_committee: Committee,
    pub encoder_committee: ShardCommittee,
    pub previous_encoder_committee: Option<ShardCommittee>,
}

/// Client for communicating with the validator node to fetch committees
pub struct EncoderValidatorClient {
    client: EncoderValidatorApiClient<Channel>,

    current_validator_committee: Committee,
    current_encoder_committee: Option<ShardCommittee>,
    previous_encoder_committee: Option<ShardCommittee>,
    current_epoch: EpochId,
}

impl EncoderValidatorClient {
    /// Create a new client that connects to the validator node at the given address
    pub async fn new(address: &Multiaddr, genesis_committee: Committee) -> Result<Self> {
        info!(
            "Creating encoder validator client connecting to {}",
            address
        );
        let channel = connect(address)
            .await
            .map_err(|err| anyhow!(err.to_string()))?;
        let client = EncoderValidatorApiClient::new(channel);

        Ok(Self {
            client,
            current_validator_committee: genesis_committee,
            current_encoder_committee: None,
            previous_encoder_committee: None,
            current_epoch: 0, // Genesis epoch
        })
    }

    /// Get the current epoch from the validator
    pub async fn get_current_epoch(&mut self) -> Result<EpochId> {
        info!("Getting current epoch from validator");

        let request = tonic::Request::new(GetLatestEpochRequest {
            start: self.current_epoch,
        });
        let response = self.client.get_latest_epoch(request).await?;

        let epoch = response.get_ref().epoch;
        info!("Current epoch reported by validator: {}", epoch);

        Ok(epoch)
    }

    /// Fetch committees for a specific epoch range
    async fn fetch_committees(
        &mut self,
        start: EpochId,
        end: EpochId,
    ) -> Result<FetchCommitteesResponse> {
        info!("Fetching committees for epochs {} to {}", start, end);

        // Ensure we're not requesting epoch 0 (genesis)
        if start == 0 {
            return Err(anyhow!(
                "Cannot request epoch 0 (genesis) committee - must be provided in configuration"
            ));
        }

        let request = tonic::Request::new(FetchCommitteesRequest { start, end });
        let response = self.client.fetch_committees(request).await?;

        debug!(
            "Received committee response with {} committees",
            response.get_ref().epoch_committees.len()
        );
        Ok(response.into_inner())
    }

    /// Convert a ValidatorSet to a Committee
    fn validator_set_to_committee(
        &self,
        validator_set: &ValidatorSet,
        epoch: EpochId,
    ) -> Result<Committee> {
        let mut voting_rights = BTreeMap::new();
        let mut authorities = BTreeMap::new();

        for (pubkey, voting_power, network_metadata) in &validator_set.0 {
            voting_rights.insert(*pubkey, *voting_power);
            authorities.insert(
                *pubkey,
                Authority {
                    stake: *voting_power,
                    address: network_metadata.consensus_address.clone(),
                    hostname: network_metadata.hostname.clone(),
                    protocol_key: network_metadata.protocol_key.clone(),
                    network_key: network_metadata.network_key.clone(),
                    authority_key: network_metadata.authority_key.clone(),
                },
            );
        }

        Ok(Committee::new(epoch, voting_rights, authorities))
    }

    fn convert_encoder_committee(
        &self,
        committee: &EncoderCommittee,
        epoch: EpochId,
    ) -> Result<ShardCommittee> {
        // Extract encoders with their voting powers
        let encoders = committee
            .members
            .iter()
            .map(|(key, voting_power)| {
                // Calculate voting power as u16
                let voting_power = (*voting_power as u16).min(10_000);

                // Create a test probe metadata (will be replaced with real data in production)
                let mut seed = [0u8; 32];
                seed[0..8].copy_from_slice(&key.to_bytes()[0..8]); // Use part of the public key as seed
                let probe = ProbeMetadata::new_for_test(&seed);

                Encoder {
                    voting_power,
                    encoder_key: key.clone(),
                    probe,
                }
            })
            .collect::<Vec<_>>();

        let shard_size = std::cmp::min(
            encoders.len() as u32,
            std::cmp::max(3, (encoders.len() / 2) as u32),
        );

        // Calculate quorum threshold - typically 2/3 rounded up
        let quorum_threshold = (shard_size * 2 + 2) / 3;

        // Create the encoder service EncoderCommittee
        Ok(ShardCommittee::new(
            epoch,
            shard_size,
            quorum_threshold,
            encoders,
        ))
    }

    /// Convert blockchain structures to our client structures
    fn convert_committees(
        &self,
        validator_set: &ValidatorSet,
        blockchain_committee: &EncoderCommittee,
        epoch: EpochId,
    ) -> Result<(Committee, ShardCommittee)> {
        let validator_committee = self.validator_set_to_committee(validator_set, epoch)?;
        let encoder_committee = self.convert_encoder_committee(blockchain_committee, epoch)?;
        Ok((validator_committee, encoder_committee))
    }

    /// Verify a single committee using a committee from the previous epoch
    fn verify_committee(
        &self,
        prev_committee: &Committee,
        committee_data: &EpochCommittee,
    ) -> Result<(Committee, ShardCommittee)> {
        info!("Verifying committee for epoch {}", committee_data.epoch);

        // Deserialize validator set and encoder committee
        let validator_set: ValidatorSet = bcs::from_bytes(&committee_data.validator_set)
            .map_err(|e| anyhow!("Failed to deserialize validator set: {}", e))?;

        let encoder_committee: EncoderCommittee =
            bcs::from_bytes(&committee_data.encoder_committee)
                .map_err(|e| anyhow!("Failed to deserialize encoder committee: {}", e))?;

        // Deserialize aggregate signatures
        let val_agg_sig: AggregateAuthoritySignature =
            bcs::from_bytes(&committee_data.aggregate_signature).map_err(|e| {
                anyhow!("Failed to deserialize validator aggregate signature: {}", e)
            })?;

        let enc_agg_sig: AggregateAuthoritySignature =
            bcs::from_bytes(&committee_data.encoder_aggregate_signature)
                .map_err(|e| anyhow!("Failed to deserialize encoder aggregate signature: {}", e))?;

        // Get the messages that were signed
        let val_digest = validator_set
            .compute_digest()
            .map_err(|e| anyhow!("Failed to compute validator set digest: {}", e))?;

        let val_message = bcs::to_bytes(&to_validator_set_intent(val_digest))
            .map_err(|e| anyhow!("Failed to serialize validator intent message: {}", e))?;

        let enc_digest = encoder_committee
            .compute_digest()
            .map_err(|e| anyhow!("Failed to compute encoder committee digest: {}", e))?;

        let enc_message = bcs::to_bytes(&to_encoder_committee_intent(enc_digest))
            .map_err(|e| anyhow!("Failed to serialize encoder intent message: {}", e))?;

        // Create a stake aggregator to check quorum
        let mut aggregator = StakeAggregator::<QuorumThreshold>::new();

        // Add each signer to the aggregator using their index
        for &index in &committee_data.signer_indices {
            let authority_index = AuthorityIndex(index);

            // Verify the signer exists in the previous committee
            if prev_committee.is_valid_index(authority_index) {
                aggregator.add(authority_index, prev_committee);
            } else {
                return Err(anyhow!(
                    "Invalid signer: authority index {} not found in previous committee",
                    authority_index
                ));
            }
        }

        // Verify we have enough stake for quorum
        if !aggregator.reached_threshold(prev_committee) {
            return Err(anyhow!(
                "Insufficient stake for quorum (got {}, needed {})",
                aggregator.stake(),
                prev_committee.quorum_threshold()
            ));
        }

        // Get public keys in same order for verification
        let pubkeys: Vec<AuthorityPublicKey> = aggregator
            .votes()
            .iter()
            .map(|&idx| {
                prev_committee
                    .authority_by_authority_index(idx)
                    .expect("Authority must exist")
                    .authority_key
                    .clone()
            })
            .collect();

        // Verify both aggregate signatures
        val_agg_sig.verify(&pubkeys, &val_message).map_err(|e| {
            warn!("Validator aggregate signature verification failed: {}", e);
            anyhow!("Invalid validator aggregate signature: {}", e)
        })?;

        enc_agg_sig.verify(&pubkeys, &enc_message).map_err(|e| {
            warn!(
                "Encoder committee aggregate signature verification failed: {}",
                e
            );
            anyhow!("Invalid encoder committee aggregate signature: {}", e)
        })?;
        // If verification succeeded, convert validator set to committee

        let validator_committee =
            self.validator_set_to_committee(&validator_set, committee_data.epoch)?;
        let encoder_committee =
            self.convert_encoder_committee(&encoder_committee, committee_data.epoch)?;

        Ok((validator_committee, encoder_committee))
    }

    /// Process committee verification in chunks up to a target epoch
    async fn verify_committee_range(
        &mut self,
        start_epoch: EpochId,
        target_epoch: EpochId,
    ) -> Result<VerifiedCommittees> {
        // Skip if we're already at or beyond the target epoch
        if self.current_epoch >= target_epoch {
            return Ok(VerifiedCommittees {
                validator_committee: self.current_validator_committee.clone(),
                encoder_committee: self.current_encoder_committee.clone().ok_or_else(|| {
                    anyhow!("No encoder committee for epoch {}", self.current_epoch)
                })?,
                previous_encoder_committee: self.previous_encoder_committee.clone(),
            });
        }

        // We'll process in manageable chunks to avoid huge requests
        const CHUNK_SIZE: u64 = 10;

        let mut current_epoch = self.current_epoch;
        let mut current_validator_committee = self.current_validator_committee.clone();
        let mut current_encoder_committee = self.current_encoder_committee.clone();
        let mut previous_encoder_committee = self.previous_encoder_committee.clone();

        // Always start from where we left off
        let mut chunk_start = start_epoch;

        while chunk_start <= target_epoch {
            // Define the end of the current chunk
            let chunk_end = std::cmp::min(chunk_start + CHUNK_SIZE - 1, target_epoch);

            info!(
                "Fetching committee chunk from {} to {}",
                chunk_start, chunk_end
            );
            let response = self.fetch_committees(chunk_start, chunk_end).await?;

            // If no committees returned and we're not at the target yet, something's wrong
            if response.epoch_committees.is_empty() {
                // This might happen if target_epoch is beyond what the validator has
                info!("No committees available for requested range");
                break;
            }

            // Sort committees by epoch to process in order
            let mut committees = response.epoch_committees;
            committees.sort_by_key(|c| c.epoch);

            // Process each committee
            for committee_data in committees {
                // Skip already processed epochs
                if committee_data.epoch <= current_epoch {
                    continue;
                }

                // Deserialize validator set and encoder committee
                let validator_set: ValidatorSet = bcs::from_bytes(&committee_data.validator_set)
                    .map_err(|e| anyhow!("Failed to deserialize validator set: {}", e))?;

                let blockchain_committee: EncoderCommittee =
                    bcs::from_bytes(&committee_data.encoder_committee)
                        .map_err(|e| anyhow!("Failed to deserialize encoder committee: {}", e))?;

                // Deserialize aggregate signatures
                let val_agg_sig: AggregateAuthoritySignature =
                    bcs::from_bytes(&committee_data.aggregate_signature).map_err(|e| {
                        anyhow!("Failed to deserialize validator aggregate signature: {}", e)
                    })?;

                let enc_agg_sig: AggregateAuthoritySignature =
                    bcs::from_bytes(&committee_data.encoder_aggregate_signature).map_err(|e| {
                        anyhow!("Failed to deserialize encoder aggregate signature: {}", e)
                    })?;

                // Get the messages that were signed
                let val_digest = validator_set
                    .compute_digest()
                    .map_err(|e| anyhow!("Failed to compute validator set digest: {}", e))?;

                let val_message = bcs::to_bytes(&to_validator_set_intent(val_digest))
                    .map_err(|e| anyhow!("Failed to serialize validator intent message: {}", e))?;

                let enc_digest = blockchain_committee
                    .compute_digest()
                    .map_err(|e| anyhow!("Failed to compute encoder committee digest: {}", e))?;

                let enc_message = bcs::to_bytes(&to_encoder_committee_intent(enc_digest))
                    .map_err(|e| anyhow!("Failed to serialize encoder intent message: {}", e))?;

                // Create a stake aggregator to check quorum
                let mut aggregator = StakeAggregator::<QuorumThreshold>::new();

                // Add each signer to the aggregator using their index
                for &index in &committee_data.signer_indices {
                    let authority_index = AuthorityIndex(index);

                    // Verify the signer exists in the previous committee
                    if current_validator_committee.is_valid_index(authority_index) {
                        aggregator.add(authority_index, &current_validator_committee);
                    } else {
                        return Err(anyhow!(
                            "Invalid signer: authority index {} not found in previous committee",
                            authority_index
                        ));
                    }
                }

                // Verify we have enough stake for quorum
                if !aggregator.reached_threshold(&current_validator_committee) {
                    return Err(anyhow!(
                        "Insufficient stake for quorum (got {}, needed {})",
                        aggregator.stake(),
                        current_validator_committee.quorum_threshold()
                    ));
                }

                // Get public keys in same order for verification
                let pubkeys: Vec<AuthorityPublicKey> = aggregator
                    .votes()
                    .iter()
                    .map(|&idx| {
                        current_validator_committee
                            .authority_by_authority_index(idx)
                            .expect("Authority must exist")
                            .authority_key
                            .clone()
                    })
                    .collect();

                // Verify both aggregate signatures
                val_agg_sig.verify(&pubkeys, &val_message).map_err(|e| {
                    warn!("Validator aggregate signature verification failed: {}", e);
                    anyhow!("Invalid validator aggregate signature: {}", e)
                })?;

                enc_agg_sig.verify(&pubkeys, &enc_message).map_err(|e| {
                    warn!(
                        "Encoder committee aggregate signature verification failed: {}",
                        e
                    );
                    anyhow!("Invalid encoder committee aggregate signature: {}", e)
                })?;

                // Convert to our client types
                let (verified_validator, verified_encoder) = self.convert_committees(
                    &validator_set,
                    &blockchain_committee,
                    committee_data.epoch,
                )?;

                // Move current to previous before updating
                if current_encoder_committee.is_some() {
                    previous_encoder_committee = current_encoder_committee;
                }

                // Update current committees
                current_validator_committee = verified_validator;
                current_encoder_committee = Some(verified_encoder);
                current_epoch = committee_data.epoch;

                info!("Verified committees for epoch {}", committee_data.epoch);
            }

            // Set up for next chunk
            chunk_start = current_epoch + 1;

            // Break if we've reached or exceeded the target
            if current_epoch >= target_epoch {
                break;
            }
        }

        // Update the client's state
        self.current_validator_committee = current_validator_committee.clone();
        self.current_encoder_committee = current_encoder_committee.clone();
        self.previous_encoder_committee = previous_encoder_committee.clone();
        self.current_epoch = current_epoch;

        Ok(VerifiedCommittees {
            validator_committee: current_validator_committee,
            encoder_committee: current_encoder_committee
                .ok_or_else(|| anyhow!("No encoder committee for epoch {}", current_epoch))?,
            previous_encoder_committee,
        })
    }

    /// Initial setup - synchronize committees from epoch 1 to current epoch
    pub async fn setup_from_genesis(&mut self) -> Result<VerifiedCommittees> {
        // First get the current epoch from the validator
        let target_epoch = self.get_current_epoch().await?;

        if target_epoch == 0 {
            // Genesis epoch - no verification needed
            return Ok(VerifiedCommittees {
                validator_committee: self.current_validator_committee.clone(),
                encoder_committee: self
                    .current_encoder_committee
                    .clone()
                    .ok_or_else(|| anyhow!("No encoder committee for epoch 0"))?,
                previous_encoder_committee: None,
            });
        }

        info!("Setting up committees from epoch 1 to {}", target_epoch);
        self.verify_committee_range(1, target_epoch).await
    }

    /// Poll for the latest committees if needed
    pub async fn poll_latest_committees(&mut self) -> Result<VerifiedCommittees> {
        // First get the current epoch from the validator
        let validator_epoch = self.get_current_epoch().await?;

        // Only sync if the validator's epoch is ahead of our current epoch
        if validator_epoch > self.current_epoch {
            info!(
                "Polling committees from epoch {} to {}",
                self.current_epoch + 1,
                validator_epoch
            );
            self.verify_committee_range(self.current_epoch + 1, validator_epoch)
                .await
        } else {
            info!("Already at latest epoch {}", self.current_epoch);
            Ok(VerifiedCommittees {
                validator_committee: self.current_validator_committee.clone(),
                encoder_committee: self
                    .current_encoder_committee
                    .clone()
                    .ok_or_else(|| anyhow!("No encoder committee available"))?,
                previous_encoder_committee: self.previous_encoder_committee.clone(),
            })
        }
    }

    /// Get the current epoch that this client knows about
    pub fn current_epoch(&self) -> EpochId {
        self.current_epoch
    }

    /// Get the current verified committees without any network calls
    pub fn current_committees(&self) -> Result<VerifiedCommittees> {
        Ok(VerifiedCommittees {
            validator_committee: self.current_validator_committee.clone(),
            encoder_committee: self
                .current_encoder_committee
                .clone()
                .ok_or_else(|| anyhow!("No encoder committee available"))?,
            previous_encoder_committee: self.previous_encoder_committee.clone(),
        })
    }
}
