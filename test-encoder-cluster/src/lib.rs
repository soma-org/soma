use std::{num::NonZeroUsize, time::Duration};

use config::EncoderConfig;
use encoder::core::encoder_node::EncoderNodeHandle;
use shared::crypto::keys::EncoderPublicKey;
use swarm::{EncoderSwarm, EncoderSwarmBuilder};
use tracing::info;

const NUM_ENCODERS: usize = 4;

#[cfg(msim)]
#[path = "./container-sim.rs"]
mod container;

mod config;
#[cfg(not(msim))]
#[path = "./container.rs"]
mod container;
mod swarm;
mod swarm_node;

pub struct TestEncoderCluster {
    pub swarm: EncoderSwarm,
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
}

pub struct TestEncoderClusterBuilder {
    num_encoders: Option<usize>,
    // genesis_config: Option<GenesisConfig>,
    // network_config: Option<NetworkConfig>,
    encoders: Option<Vec<EncoderConfig>>,
}

impl TestEncoderClusterBuilder {
    pub fn new() -> Self {
        TestEncoderClusterBuilder {
            num_encoders: None,
            // genesis_config: None,
            // network_config: None,
            encoders: None,
        }
    }

    pub fn with_num_encoders(mut self, num: usize) -> Self {
        self.num_encoders = Some(num);
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
        let swarm = self.start_swarm().await.unwrap();

        TestEncoderCluster { swarm }
    }

    async fn start_swarm(&mut self) -> Result<EncoderSwarm, anyhow::Error> {
        let mut builder: EncoderSwarmBuilder = EncoderSwarm::builder();

        if let Some(encoders) = self.encoders.take() {
            builder = builder.with_encoders(encoders);
        } else {
            builder = builder.committee_size(
                NonZeroUsize::new(self.num_encoders.unwrap_or(NUM_ENCODERS)).unwrap(),
            )
        };

        let mut swarm = builder.build();
        swarm.launch().await?;
        Ok(swarm)
    }
}

impl Default for TestEncoderClusterBuilder {
    fn default() -> Self {
        Self::new()
    }
}
