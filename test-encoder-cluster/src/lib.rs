use std::{sync::Arc, time::Duration};

use encoder::core::encoder_node::EncoderNodeHandle;
use object_store::memory::InMemory;
use rand::{rngs::StdRng, SeedableRng};

use swarm::EncoderSwarm;
use tracing::info;
use types::crypto::NetworkKeyPair;
use types::parameters::TonicParameters;
use types::shard_crypto::keys::EncoderPublicKey;
use types::{base::SomaAddress, config::encoder_config::EncoderConfig};

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
    pub client_keypair: NetworkKeyPair,
    pub parameters: Arc<TonicParameters>,
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
}

pub struct TestEncoderClusterBuilder {
    encoders: Option<Vec<EncoderConfig>>,
    client_keypair: Option<NetworkKeyPair>,
    shared_object_store: Option<Arc<InMemory>>,
}

impl TestEncoderClusterBuilder {
    pub fn new() -> Self {
        TestEncoderClusterBuilder {
            encoders: None,
            client_keypair: None,
            shared_object_store: None,
        }
    }

    pub fn with_encoders(mut self, encoders: Vec<EncoderConfig>) -> Self {
        self.encoders = Some(encoders);
        self
    }

    pub fn with_client_keypair(mut self, keypair: NetworkKeyPair) -> Self {
        self.client_keypair = Some(keypair);
        self
    }

    pub fn with_shared_object_store(mut self, store: Arc<InMemory>) -> Self {
        self.shared_object_store = Some(store);
        self
    }

    pub async fn build(mut self) -> TestEncoderCluster {
        // Generate client keypair internally if not provided
        let client_keypair = self.client_keypair.unwrap_or_else(|| {
            let mut rng = StdRng::from_seed([0; 32]); // Use deterministic seed for testing
            NetworkKeyPair::generate(&mut rng)
        });

        let mut builder = EncoderSwarm::builder();

        if let Some(encoders) = self.encoders {
            builder = builder.with_encoders(encoders);
        } else {
            panic!("Encoder configs must be provided when building TestEncoderCluster");
        }

        if let Some(store) = self.shared_object_store {
            builder = builder.with_shared_object_store(store);
        }

        let mut swarm = builder.build();
        let _ = swarm.launch().await;

        TestEncoderCluster {
            swarm,
            client_keypair,
            parameters: Arc::new(TonicParameters::default()),
        }
    }
}

impl Default for TestEncoderClusterBuilder {
    fn default() -> Self {
        Self::new()
    }
}
