use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc, time::Duration};

use bytes::Bytes;
use config::EncoderConfig;
use encoder::{
    core::encoder_node::EncoderNodeHandle,
    error::{ShardError, ShardResult},
    messaging::{
        tonic::{
            external::{EncoderExternalTonicClient, SendInputRequest},
            NetworkingInfo,
        },
        EncoderExternalNetworkClient,
    },
    types::{
        parameters::Parameters,
        shard::{Shard, ShardEntropy},
        shard_input::{ShardInput, ShardInputV1},
        shard_verifier::ShardAuthToken,
    },
};
use fastcrypto::{
    bls12381::min_sig::{self, BLS12381PublicKey},
    traits::KeyPair,
};
use rand::{rngs::StdRng, SeedableRng};
use shared::{
    crypto::keys::{EncoderKeyPair, EncoderPublicKey, PeerKeyPair},
    digest::Digest,
    entropy::{BlockEntropy, BlockEntropyProof},
    scope::Scope,
    signed::Signed,
    verified::Verified,
};
use swarm::{multiaddr_compat::to_network_multiaddr, EncoderSwarm, EncoderSwarmBuilder};
use tonic::Request;
use tracing::info;
use vdf::{
    class_group::{discriminant::DISCRIMINANT_3072, QuadraticForm},
    vdf::{wesolowski::DefaultVDF, VDF},
};

const NUM_ENCODERS: usize = 4;

pub mod config;
#[cfg(msim)]
#[path = "./container-sim.rs"]
mod container;
#[cfg(not(msim))]
#[path = "./container.rs"]
mod container;
pub mod swarm;
mod swarm_node;

pub struct TestEncoderCluster {
    pub swarm: EncoderSwarm,
    pub client_keypair: PeerKeyPair,
    pub parameters: Arc<Parameters>,
}

impl TestEncoderCluster {
    pub fn all_encoder_handles(&self) -> Vec<EncoderNodeHandle> {
        self.swarm
            .encoder_nodes()
            .flat_map(|n| n.get_node_handle())
            .collect()
    }

    pub fn get_encoder_pubkeys(&self) -> Vec<EncoderPublicKey> {
        self.swarm.active_encoders().map(|v| v.name()).collect()
    }

    pub fn stop_node(&self, name: &EncoderPublicKey) {
        self.swarm.node(name).unwrap().stop();
    }

    pub async fn stop_all_encoders(&self) {
        info!("Stopping all encoders in the cluster");
        self.swarm.active_encoders().for_each(|v| v.stop());
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    pub async fn start_all_encoders(&self) {
        info!("Starting all encoders in the cluster");
        for v in self.swarm.encoder_nodes() {
            if v.is_running() {
                continue;
            }
            v.start().await.unwrap();
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    pub async fn start_node(&self, name: &EncoderPublicKey) {
        let node = self.swarm.node(name).unwrap();
        if node.is_running() {
            return;
        }
        node.start().await.unwrap();
    }

    pub async fn spawn_new_encoder(&mut self, config: EncoderConfig) -> EncoderNodeHandle {
        // TODO: when spawning a new encoder, make sure that the config is updated to know of the current encoder committee
        // let seed_peers = self
        //     .swarm
        //     .config()
        //     .validator_configs
        //     .iter()
        //     .map(|config| SeedPeer {
        //         peer_id: Some(PeerId(
        //             config.network_key_pair().public().into_inner().0.to_bytes(),
        //         )),
        //         address: config.p2p_config.external_address.clone().unwrap(),
        //     })
        //     .collect();

        self.swarm.spawn_new_encoder(config).await
    }
    // Get a preconfigured external client
    pub fn get_external_client(&self) -> EncoderExternalTonicClient {
        EncoderExternalTonicClient::new(
            self.swarm.external_addresses.clone(),
            self.client_keypair.clone(),
            self.parameters.clone(),
            100,
        )
    }

    // Generate a deterministic EncoderKeyPair derived from the client's PeerKeyPair
    fn get_client_encoder_keypair(&self) -> EncoderKeyPair {
        // Create a seed from the client's PeerKeyPair's Ed25519 private key bytes
        let seed = self.client_keypair.clone().private_key_bytes();

        // Create a deterministic RNG from the seed
        let mut rng = rand::rngs::StdRng::from_seed(
            seed.try_into().unwrap_or([0; 32]), // Convert to fixed-size array
        );

        // Generate a deterministic EncoderKeyPair
        EncoderKeyPair::generate(&mut rng)
    }
    // Create a proper ShardInput with valid VDF entropy
    pub fn create_valid_shard_input(&self) -> ShardInput {
        let auth_token = create_valid_test_token();
        ShardInput::V1(ShardInputV1::new(auth_token))
    }

    // Update the existing method to use the valid input
    pub fn create_signed_input(&self) -> Verified<Signed<ShardInput, min_sig::BLS12381Signature>> {
        // Get a properly formed ShardInput with valid AuthToken
        let input = self.create_valid_shard_input();

        // Get a keypair for signing
        let encoder_keypair = self.get_client_encoder_keypair();
        let inner_keypair = encoder_keypair.inner().copy();

        // Sign the input
        let signed_input = Signed::new(input, Scope::ShardInput, &inner_keypair.private()).unwrap();

        // Create a verified object
        Verified::from_trusted(signed_input).unwrap()
    }

    pub fn get_shard_from_token(&self, token: &ShardAuthToken) -> ShardResult<Shard> {
        // Get the first encoder node to access its context and committee
        let any_encoder = self
            .swarm
            .encoder_nodes()
            .next()
            .ok_or_else(|| ShardError::NotFound("No encoder nodes available".to_string()))?;

        if let Some(handle) = any_encoder.get_node_handle() {
            // Use the encoder's context to access the committee
            let context = handle.with(|node| node.context.clone());

            // Create the shard entropy from the token
            let shard_entropy_input = ShardEntropy::new(
                token.metadata_commitment.clone(),
                token.block_entropy.clone(),
            );

            // Create a digest from the shard entropy
            let shard_entropy = Digest::new(&shard_entropy_input).unwrap();

            // Get the committees from the context
            let inner_context = context.inner();
            let committees = inner_context.committees(token.epoch())?;

            // Sample the shard using the same method as ShardVerifier
            let shard = committees.encoder_committee.sample_shard(shard_entropy)?;

            Ok(shard)
        } else {
            Err(ShardError::NotFound(
                "No active encoder node handle available".to_string(),
            ))
        }
    }

    pub async fn send_to_shard_members(
        &self,
        token: &ShardAuthToken,
        timeout: Duration,
    ) -> Result<(), Vec<(EncoderPublicKey, ShardError)>> {
        // Get the shard from the token
        let shard = self.get_shard_from_token(token).unwrap();

        // Get the encoder public keys in the shard
        let shard_members = shard.encoders();

        // Create the input with the given token
        let input = self.create_signed_input_with_token(token.clone());

        // Send input only to shard members
        let mut errors = Vec::new();
        let client = self.get_external_client();

        for key in shard_members {
            match client.send_input(&key, &input, timeout).await {
                Ok(_) => {}
                Err(e) => errors.push((key, e)),
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn create_signed_input_with_token(
        &self,
        token: ShardAuthToken,
    ) -> Verified<Signed<ShardInput, min_sig::BLS12381Signature>> {
        // Create ShardInput with the provided token
        let input = ShardInput::V1(ShardInputV1::new(token));

        // Get a keypair for signing
        let encoder_keypair = self.get_client_encoder_keypair();
        let inner_keypair = encoder_keypair.inner().copy();

        // Sign the input
        let signed_input = Signed::new(input, Scope::ShardInput, &inner_keypair.private()).unwrap();

        // Create a verified object
        Verified::from_trusted(signed_input).unwrap()
    }
}

pub fn create_valid_test_token() -> ShardAuthToken {
    // Reuse the existing token creation code
    let basic_token = ShardAuthToken::new_for_test();
    let epoch = basic_token.epoch();
    let block_ref = basic_token.proof.block_ref();

    // Create VDF entropy and proof directly without relying on the cache
    // Generate a seed for VDF input
    let seed = bcs::to_bytes(&(epoch, block_ref)).unwrap();

    // Create input for VDF
    let input =
        QuadraticForm::hash_to_group_with_default_parameters(&seed, &DISCRIMINANT_3072).unwrap();

    // Create VDF instance directly (bypass the cache)
    let iterations = 1; // Use minimal iterations for testing
    let vdf = DefaultVDF::new(DISCRIMINANT_3072.clone(), iterations);

    // Evaluate VDF directly
    let (output, proof) = vdf.evaluate(&input).unwrap();

    // Convert output and proof to the expected format
    let entropy_bytes = bcs::to_bytes(&output).unwrap();
    let proof_bytes = bcs::to_bytes(&proof).unwrap();

    let block_entropy = BlockEntropy::new(Bytes::copy_from_slice(&entropy_bytes));
    let entropy_proof = BlockEntropyProof::new(Bytes::copy_from_slice(&proof_bytes));

    // Create a new token with valid entropy data
    ShardAuthToken {
        proof: basic_token.proof,
        metadata_commitment: basic_token.metadata_commitment,
        block_entropy,
        entropy_proof,
    }
}

pub struct TestEncoderClusterBuilder {
    num_encoders: Option<usize>,
    // genesis_config: Option<GenesisConfig>,
    // network_config: Option<NetworkConfig>,
    encoders: Option<Vec<EncoderConfig>>,
    client_keypair: Option<PeerKeyPair>, // Add this field
}

impl TestEncoderClusterBuilder {
    pub fn new() -> Self {
        TestEncoderClusterBuilder {
            num_encoders: None,
            // genesis_config: None,
            // network_config: None,
            encoders: None,
            client_keypair: None,
        }
    }

    pub fn with_num_encoders(mut self, num: usize) -> Self {
        self.num_encoders = Some(num);
        self
    }

    pub fn with_client_keypair(mut self, keypair: PeerKeyPair) -> Self {
        self.client_keypair = Some(keypair);
        self
    }

    // pub fn set_genesis_config(mut self, genesis_config: GenesisConfig) -> Self {
    //     assert!(self.genesis_config.is_none() && self.network_config.is_none());
    //     self.genesis_config = Some(genesis_config);
    //     self
    // }

    // pub fn set_network_config(mut self, network_config: NetworkConfig) -> Self {
    //     assert!(self.genesis_config.is_none() && self.network_config.is_none());
    //     self.network_config = Some(network_config);
    //     self
    // }

    /// Provide encoder configs, overrides the `num_encoders` setting.
    pub fn with_encoders(mut self, encoders: Vec<EncoderConfig>) -> Self {
        self.encoders = Some(encoders);
        self
    }

    pub async fn build(mut self) -> TestEncoderCluster {
        // Generate client keypair internally if not provided
        let client_keypair = self.client_keypair.unwrap_or_else(|| {
            let mut rng = StdRng::from_seed([0; 32]); // Use deterministic seed for testing
            PeerKeyPair::generate(&mut rng)
        });

        let mut builder = EncoderSwarm::builder().with_client_keypair(client_keypair.clone());

        if let Some(encoders) = self.encoders.take() {
            builder = builder.with_encoders(encoders);
        } else {
            builder = builder.committee_size(
                NonZeroUsize::new(self.num_encoders.unwrap_or(NUM_ENCODERS)).unwrap(),
            );
        }

        let mut swarm = builder.build();
        let _ = swarm.launch().await;

        TestEncoderCluster {
            swarm,
            client_keypair,
            parameters: Arc::new(Parameters::default()),
        }
    }
}

impl Default for TestEncoderClusterBuilder {
    fn default() -> Self {
        Self::new()
    }
}
