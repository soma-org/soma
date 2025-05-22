use anyhow::{anyhow, Result};
use encoder_validator_api::tonic_gen::encoder_validator_api_client::EncoderValidatorApiClient;
use shared::encoder_committee::{Encoder, EncoderCommittee as ShardCommittee};
use shared::{
    authority_committee::AuthorityCommittee,
    crypto::keys::{
        AuthorityPublicKey as SharedAuthorityPublicKey, EncoderPublicKey, PeerPublicKey,
        ProtocolPublicKey,
    },
    probe::ProbeMetadata,
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
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

pub struct VerifiedCommittees {
    pub validator_committee: Committee,
    pub encoder_committee: ShardCommittee,
    pub previous_encoder_committee: Option<ShardCommittee>,
}

/// Enriched result type with all necessary committee information
pub struct EnrichedVerifiedCommittees {
    // Original verification data
    pub validator_committee: Committee,
    pub encoder_committee: ShardCommittee,
    pub previous_encoder_committee: Option<ShardCommittee>,

    // Additional data for CommitteeSyncManager
    pub authority_committee: AuthorityCommittee,
    pub networking_info:
        BTreeMap<EncoderPublicKey, (soma_network::multiaddr::Multiaddr, PeerPublicKey)>,
    pub connections_info: BTreeMap<PeerPublicKey, EncoderPublicKey>,
    pub object_servers:
        HashMap<EncoderPublicKey, (PeerPublicKey, soma_network::multiaddr::Multiaddr)>,
    pub epoch_start_timestamp_ms: u64,
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

    /// Convert blockchain structures to our client structures
    fn convert_committees(
        &self,
        validator_set: &ValidatorSet,
        blockchain_committee: &EncoderCommittee,
        epoch: EpochId,
    ) -> Result<(Committee, ShardCommittee)> {
        let validator_committee = self.validator_set_to_committee(validator_set, epoch)?;
        let encoder_committee =
            EncoderCommittee::convert_encoder_committee(blockchain_committee, epoch);
        Ok((validator_committee, encoder_committee))
    }

    pub fn extract_network_info(
        encoder_committee: &types::committee::EncoderCommittee,
        previous_encoder_committee: Option<&types::committee::EncoderCommittee>,
    ) -> (
        BTreeMap<EncoderPublicKey, (soma_network::multiaddr::Multiaddr, PeerPublicKey)>, // For NetworkingInfo
        BTreeMap<PeerPublicKey, EncoderPublicKey>, // For ConnectionsInfo
        HashMap<EncoderPublicKey, (PeerPublicKey, soma_network::multiaddr::Multiaddr)>, // For object servers
    ) {
        let mut networking_info = BTreeMap::new();
        let mut connections_info = BTreeMap::new();
        let mut object_servers = HashMap::new();

        // Helper function to process a single committee
        let process_committee = |committee: &types::committee::EncoderCommittee,
                                 networking: &mut BTreeMap<_, _>,
                                 connections: &mut BTreeMap<_, _>,
                                 objects: &mut HashMap<_, _>| {
            // Process each encoder and its network metadata
            for (encoder_key, _) in &committee.members {
                if let Some(metadata) = committee.network_metadata.get(encoder_key) {
                    // Convert NetworkPublicKey to PeerPublicKey (they have the same inner type)
                    let peer_key = PeerPublicKey::new(metadata.network_key.clone().into_inner());

                    // Add to network info mapping
                    networking.insert(
                        encoder_key.clone(),
                        (
                            metadata
                                .network_address
                                .clone()
                                .to_string()
                                .parse()
                                .expect("Valid multiaddr"),
                            peer_key.clone(),
                        ),
                    );

                    // Add to connections info mapping
                    connections.insert(peer_key.clone(), encoder_key.clone());

                    objects.insert(
                        encoder_key.clone(),
                        (
                            peer_key,
                            metadata
                                .object_server_address
                                .clone()
                                .to_string()
                                .parse()
                                .expect("Valid multiaddr"),
                        ),
                    );
                }
            }
        };

        // First process previous committee (so current can override if needed)
        if let Some(prev) = previous_encoder_committee {
            process_committee(
                prev,
                &mut networking_info,
                &mut connections_info,
                &mut object_servers,
            );
        }

        // Then process current committee
        process_committee(
            encoder_committee,
            &mut networking_info,
            &mut connections_info,
            &mut object_servers,
        );

        (networking_info, connections_info, object_servers)
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
            EncoderCommittee::convert_encoder_committee(&encoder_committee, committee_data.epoch);

        Ok((validator_committee, encoder_committee))
    }

    /// Verify a single committee and enrich it with network data
    fn enrich_committee(
        &self,
        validator_committee: Committee,
        encoder_committee: ShardCommittee,
        previous_encoder_committee: Option<ShardCommittee>,
        committee_data: &EpochCommittee,
        previous_committee_data: Option<&EpochCommittee>,
    ) -> Result<EnrichedVerifiedCommittees> {
        // Create authority committee
        let authority_committee = Committee::convert_to_authority_committee(&validator_committee);

        // Extract timestamp
        let epoch_start_timestamp_ms = committee_data.next_epoch_start_timestamp_ms;

        // Extract blockchain committee for network info
        let blockchain_committee: EncoderCommittee =
            bcs::from_bytes(&committee_data.encoder_committee)
                .map_err(|e| anyhow!("Failed to deserialize encoder committee: {}", e))?;

        // Extract previous blockchain committee if available
        let previous_blockchain_committee = if let Some(prev_data) = previous_committee_data {
            match bcs::from_bytes::<EncoderCommittee>(&prev_data.encoder_committee) {
                Ok(committee) => Some(committee),
                Err(e) => {
                    warn!("Failed to deserialize previous encoder committee: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Extract network info
        let (networking_info, connections_info, object_servers) =
            EncoderValidatorClient::extract_network_info(
                &blockchain_committee,
                previous_blockchain_committee.as_ref(),
            );

        Ok(EnrichedVerifiedCommittees {
            validator_committee,
            encoder_committee,
            previous_encoder_committee,
            authority_committee,
            networking_info,
            connections_info,
            object_servers,
            epoch_start_timestamp_ms,
        })
    }

    /// Process committee verification in chunks up to a target epoch
    async fn verify_committee_range(
        &mut self,
        start_epoch: EpochId,
        target_epoch: EpochId,
    ) -> Result<EnrichedVerifiedCommittees> {
        // Skip if we're already at or beyond the target epoch
        if self.current_epoch >= target_epoch {
            let enriched = EnrichedVerifiedCommittees {
                validator_committee: self.current_validator_committee.clone(),
                encoder_committee: self.current_encoder_committee.clone().ok_or_else(|| {
                    anyhow!("No encoder committee for epoch {}", self.current_epoch)
                })?,
                previous_encoder_committee: self.previous_encoder_committee.clone(),
                authority_committee: Committee::convert_to_authority_committee(
                    &self.current_validator_committee,
                ),
                networking_info: BTreeMap::new(),
                connections_info: BTreeMap::new(),
                object_servers: HashMap::new(),
                epoch_start_timestamp_ms: 0, // No new epoch data
            };
            return Ok(enriched);
        }

        // We'll process in manageable chunks to avoid huge requests
        const CHUNK_SIZE: u64 = 10;

        let mut current_epoch = self.current_epoch;
        let mut current_validator_committee = self.current_validator_committee.clone();
        let mut current_encoder_committee = self.current_encoder_committee.clone();
        let mut previous_encoder_committee = self.previous_encoder_committee.clone();

        // Track the most recently processed committee data for enrichment
        let mut last_committee_data: Option<EpochCommittee> = None;
        let mut previous_committee_data: Option<EpochCommittee> = None;

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

                // Use verify_committee for each committee
                let (validator_committee, encoder_committee) =
                    self.verify_committee(&current_validator_committee, &committee_data)?;

                // Store the committee data for previous epoch
                if current_encoder_committee.is_some() {
                    previous_encoder_committee = current_encoder_committee;
                    if let Some(last_data) = last_committee_data.take() {
                        previous_committee_data = Some(last_data);
                    }
                }

                // Update current committees
                current_validator_committee = validator_committee;
                current_encoder_committee = Some(encoder_committee);
                current_epoch = committee_data.epoch;

                // Keep track of the most recent committee data
                last_committee_data = Some(committee_data);

                info!("Verified committees for epoch {}", current_epoch);
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

        // Create enriched result using committee data
        if let (Some(encoder_committee), Some(committee_data)) =
            (current_encoder_committee, last_committee_data)
        {
            // Create enriched result using both current and previous committee data
            let enriched = self.enrich_committee(
                current_validator_committee,
                encoder_committee,
                previous_encoder_committee,
                &committee_data,
                previous_committee_data.as_ref(),
            )?;

            Ok(enriched)
        } else {
            // If no committees were processed or we don't have an encoder committee
            Err(anyhow!("No valid committees found or processed"))
        }
    }

    /// Initial setup - synchronize committees from epoch 1 to current epoch
    pub async fn setup_from_genesis(&mut self) -> Result<EnrichedVerifiedCommittees> {
        // First get the current epoch from the validator
        let target_epoch = self.get_current_epoch().await?;

        if target_epoch == 0 {
            // Genesis epoch - create empty enriched result
            return Err(anyhow!("Cannot enrich genesis committee"));
        }

        info!("Setting up committees from epoch 1 to {}", target_epoch);
        self.verify_committee_range(1, target_epoch).await
    }

    /// Poll for the latest committees if needed
    pub async fn poll_latest_committees(&mut self) -> Result<EnrichedVerifiedCommittees> {
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

            Err(anyhow!("Already at latest epoch"))
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
