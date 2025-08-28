use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc, time::Duration};

use bytes::Bytes;
use encoder::core::encoder_node::EncoderNodeHandle;
use fastcrypto::{
    bls12381::min_sig::{self, BLS12381PublicKey},
    traits::KeyPair,
};
use rand::{rngs::StdRng, SeedableRng};

use shared::parameters::Parameters;
use shared::{
    crypto::keys::{EncoderKeyPair, EncoderPublicKey, PeerKeyPair},
    digest::Digest,
    entropy::{BlockEntropy, BlockEntropyProof},
    error::{ShardError, ShardResult},
    scope::Scope,
    shard::{Shard, ShardEntropy},
    signed::Signed,
    verified::Verified,
};
use swarm::{EncoderSwarm, EncoderSwarmBuilder};
use tonic::Request;
use tracing::info;
use types::shard::{ShardAuthToken, ShardInput, ShardInputV1};
use types::{base::SomaAddress, config::encoder_config::EncoderConfig};
use vdf::{
    class_group::{discriminant::DISCRIMINANT_3072, QuadraticForm},
    vdf::{wesolowski::DefaultVDF, VDF},
};

const NUM_ENCODERS: usize = 4;

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

    // Spawn a new encoder node from a config
    pub async fn spawn_new_encoder(&mut self, config: EncoderConfig) -> EncoderNodeHandle {
        self.swarm.spawn_new_encoder(config).await
    }

    // Get encoder address from config
    pub fn get_address_from_config(config: &EncoderConfig) -> SomaAddress {
        (&config.account_keypair.keypair().public()).into()
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
}

pub struct TestEncoderClusterBuilder {
    encoders: Option<Vec<EncoderConfig>>,
    client_keypair: Option<PeerKeyPair>,
}

impl TestEncoderClusterBuilder {
    pub fn new() -> Self {
        TestEncoderClusterBuilder {
            encoders: None,
            client_keypair: None,
        }
    }

    pub fn with_encoders(mut self, encoders: Vec<EncoderConfig>) -> Self {
        self.encoders = Some(encoders);
        self
    }

    pub fn with_client_keypair(mut self, keypair: PeerKeyPair) -> Self {
        self.client_keypair = Some(keypair);
        self
    }

    pub async fn build(mut self) -> TestEncoderCluster {
        // Generate client keypair internally if not provided
        let client_keypair = self.client_keypair.unwrap_or_else(|| {
            let mut rng = StdRng::from_seed([0; 32]); // Use deterministic seed for testing
            PeerKeyPair::generate(&mut rng)
        });

        let mut builder = EncoderSwarm::builder().with_client_key(client_keypair.public());

        if let Some(encoders) = self.encoders {
            builder = builder.with_encoders(encoders);
        } else {
            panic!("Encoder configs must be provided when building TestEncoderCluster");
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
