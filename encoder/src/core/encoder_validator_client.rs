// src/encoder_validator_client.rs

use anyhow::{anyhow, Result};
use encoder_validator_api::tonic_gen::encoder_validator_api_client::EncoderValidatorApiClient;
use shared::crypto::keys::EncoderPublicKey;
use std::{collections::BTreeMap, sync::Arc};
use tonic::transport::{Channel, Endpoint};
use tracing::{debug, info, warn};
use types::{
    base::AuthorityName,
    client::connect,
    committee::{Authority, AuthorityIndex, Committee, EpochId},
    consensus::{
        stake_aggregator::{QuorumThreshold, StakeAggregator},
        validator_set::{to_validator_set_intent, ValidatorSet},
    },
    crypto::{AggregateAuthenticator, AggregateAuthoritySignature, AuthorityPublicKey},
    encoder_validator::{EpochCommittee, FetchCommitteesRequest, FetchCommitteesResponse},
    multiaddr::Multiaddr,
};

/// Client for communicating with the validator node to fetch committees
pub struct EncoderValidatorClient {
    client: EncoderValidatorApiClient<Channel>,
    genesis_committee: Committee,
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
            genesis_committee,
        })
    }

    /// Fetch committees for the given range of epochs
    pub async fn fetch_committees(
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

    /// Verify a single committee using a committee from the previous epoch
    fn verify_committee(
        &self,
        prev_committee: &Committee,
        committee_data: &EpochCommittee,
    ) -> Result<Committee> {
        info!("Verifying committee for epoch {}", committee_data.epoch);

        // Deserialize validator set
        let validator_set: ValidatorSet = bcs::from_bytes(&committee_data.validator_set)
            .map_err(|e| anyhow!("Failed to deserialize validator set: {}", e))?;

        // Deserialize aggregate signature
        let agg_sig: AggregateAuthoritySignature =
            bcs::from_bytes(&committee_data.aggregate_signature)
                .map_err(|e| anyhow!("Failed to deserialize aggregate signature: {}", e))?;

        // Get the message that was signed (validator set intent)
        let digest = validator_set
            .compute_digest()
            .map_err(|e| anyhow!("Failed to compute validator set digest: {}", e))?;

        let message = bcs::to_bytes(&to_validator_set_intent(digest))
            .map_err(|e| anyhow!("Failed to serialize intent message: {}", e))?;

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

        // Verify the aggregate signature using the validator set message
        agg_sig.verify(&pubkeys, &message).map_err(|e| {
            warn!("Aggregate signature verification failed: {}", e);
            anyhow!("Invalid aggregate signature: {}", e)
        })?;

        // If verification succeeded, convert validator set to committee
        self.validator_set_to_committee(&validator_set, committee_data.epoch)
    }

    /// Verify the committee response using the chain of signatures
    pub fn verify_committees(&self, response: &FetchCommitteesResponse) -> Result<Vec<Committee>> {
        info!("Verifying committee response");

        // Start with the genesis committee
        let mut current_committee = self.genesis_committee.clone();
        let mut verified_committees = Vec::new();

        // Sort committees by epoch to ensure we verify in the correct order
        let mut committees = response.epoch_committees.clone();
        committees.sort_by_key(|c| c.epoch);

        for committee in committees {
            // Skip verification if we're missing data
            if committee.validator_set.is_empty() || committee.aggregate_signature.is_empty() {
                return Err(anyhow!(
                    "Missing validator set or signature for epoch {}",
                    committee.epoch
                ));
            }

            // Verify the committee using the current committee (from previous epoch)
            let verified_committee = self.verify_committee(&current_committee, &committee)?;

            // Save the verified committee
            verified_committees.push(verified_committee.clone());

            // Update current committee for next iteration
            current_committee = verified_committee;
        }

        info!("Committee verification successful");
        Ok(verified_committees)
    }

    /// Fetch and verify committees for the given range
    pub async fn fetch_and_verify_committees(
        &mut self,
        start: EpochId,
        end: EpochId,
    ) -> Result<Vec<Committee>> {
        let response = self.fetch_committees(start, end).await?;
        self.verify_committees(&response)
    }
}
