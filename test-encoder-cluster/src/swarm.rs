use anyhow::Result;
use encoder::{
    core::encoder_node::EncoderNodeHandle,
    messaging::tonic::{internal::ConnectionsInfo, NetworkingInfo},
};
use futures::future::try_join_all;
use rand::{
    rngs::{OsRng, StdRng},
    SeedableRng,
};
use shared::crypto::keys::{EncoderKeyPair, EncoderPublicKey, PeerKeyPair};
use soma_tls::AllowPublicKeys;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    net::{IpAddr, Ipv4Addr},
    num::NonZeroUsize,
    path::PathBuf,
};
use tracing::info;

use crate::{
    config::{EncoderCommitteeConfig, EncoderConfig},
    swarm_node::Node,
};

#[derive(Debug)]
pub struct EncoderSwarm {
    nodes: HashMap<EncoderPublicKey, Node>,
}

impl Drop for EncoderSwarm {
    fn drop(&mut self) {
        self.nodes_iter_mut().for_each(|node| node.stop());
    }
}

impl EncoderSwarm {
    fn nodes_iter_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.nodes.values_mut()
    }

    /// Return a new Builder
    pub fn builder() -> EncoderSwarmBuilder {
        EncoderSwarmBuilder::new()
    }

    /// Start all nodes associated with this Swarm
    pub async fn launch(&mut self) -> Result<()> {
        try_join_all(self.nodes_iter_mut().map(|node| node.start())).await?;
        tracing::info!("Successfully launched Encoder Swarm");
        Ok(())
    }

    // Doesn't mean all the encoders are active
    pub fn encoder_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn node(&self, name: &EncoderPublicKey) -> Option<&Node> {
        self.nodes.get(name)
    }

    pub fn encoder_node_handles(&self) -> Vec<EncoderNodeHandle> {
        self.encoder_nodes()
            .map(|node| node.get_node_handle().unwrap())
            .collect()
    }

    /// Returns an iterator over all currently active encoders.
    pub fn active_encoders(&self) -> impl Iterator<Item = &Node> {
        self.encoder_nodes().filter(|node| {
            node.get_node_handle().map_or(false, |handle| {
                // TODO: define what an active encoder is by letting EncoderNode's state be inspectable
                true
                // let state = handle.state();
                // state.is_active()
            })
        })
    }

    pub async fn spawn_new_encoder(&mut self, config: EncoderConfig) -> EncoderNodeHandle {
        let name = config.protocol_public_key();
        let node = Node::new(config);
        node.start().await.unwrap();
        let handle = node.get_node_handle().unwrap();
        self.nodes.insert(name, node);
        handle
    }
}

pub struct EncoderSwarmBuilder<R = StdRng> {
    rng: R,
    committee: EncoderCommitteeConfig,
}

impl EncoderSwarmBuilder {
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_seed([0; 32]), // TODO: change this
            committee: EncoderCommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
        }
    }
}

/// Implementation of EncoderSwarmBuilder to work with EncoderCommitteeConfig
impl<R> EncoderSwarmBuilder<R> {
    pub fn committee(mut self, committee: EncoderCommitteeConfig) -> Self {
        self.committee = committee;
        self
    }

    pub fn committee_size(mut self, committee_size: NonZeroUsize) -> Self {
        self.committee = EncoderCommitteeConfig::Size(committee_size);
        self
    }

    pub fn with_encoders(mut self, encoders: Vec<EncoderConfig>) -> Self {
        self.committee = EncoderCommitteeConfig::Encoders(encoders);
        self
    }
}

/// Modified EncoderSwarmBuilder to work with EncoderCommitteeConfig
impl<R: rand::RngCore + rand::CryptoRng + fastcrypto::traits::AllowedRng> EncoderSwarmBuilder<R> {
    pub fn build(mut self) -> EncoderSwarm {
        // Determine the number of encoders to create based on the committee config
        let mut encoder_configs = match self.committee {
            EncoderCommitteeConfig::Size(size) => {
                // Generate new encoder configs
                let mut configs = Vec::with_capacity(size.get());
                for i in 0..size.get() {
                    // Generate keypairs
                    let encoder_keypair = EncoderKeyPair::generate(&mut self.rng);
                    let peer_keypair = PeerKeyPair::generate(&mut self.rng);

                    // Generate unique ports for each node
                    let encoder_port = 9000 + (i as u16 * 2);
                    let object_port = 9001 + (i as u16 * 2);

                    let project_root = PathBuf::from("/tmp"); // Default test paths
                    let entry_point = PathBuf::from("test_module.py");

                    configs.push(EncoderConfig::new(
                        encoder_keypair,
                        peer_keypair,
                        IpAddr::V4(Ipv4Addr::LOCALHOST),
                        encoder_port,
                        object_port,
                        project_root,
                        entry_point,
                    ));
                }
                configs
            }
            EncoderCommitteeConfig::Encoders(encoders) => {
                // Use provided encoder configs
                encoders
            }
            EncoderCommitteeConfig::EncoderKeys(keys) => {
                // Generate new encoder configs using provided keypairs
                let mut configs = Vec::with_capacity(keys.len());
                for (i, key) in keys.into_iter().enumerate() {
                    // Generate peer keypair
                    let peer_keypair = PeerKeyPair::generate(&mut self.rng);

                    // Generate unique ports for each node
                    let encoder_port = 9000 + (i as u16 * 2);
                    let object_port = 9001 + (i as u16 * 2);

                    let project_root = PathBuf::from("/tmp");
                    let entry_point = PathBuf::from("test_module.py");

                    configs.push(EncoderConfig::new(
                        key,
                        peer_keypair,
                        IpAddr::V4(Ipv4Addr::LOCALHOST),
                        encoder_port,
                        object_port,
                        project_root,
                        entry_point,
                    ));
                }
                configs
            }
            EncoderCommitteeConfig::Deterministic((size, keys_opt)) => {
                let keys = keys_opt.unwrap_or_else(|| {
                    (0..size.get())
                        .map(|_| EncoderKeyPair::generate(&mut self.rng))
                        .collect()
                });

                // Generate deterministic configs
                let mut configs = Vec::with_capacity(keys.len());
                for (i, key) in keys.into_iter().enumerate() {
                    // Generate peer keypair
                    let peer_keypair = PeerKeyPair::generate(&mut self.rng);

                    let port_offset = 8000 + i * 10;
                    let encoder_port = port_offset as u16;
                    let object_port = (port_offset + 1) as u16;

                    let project_root = PathBuf::from("/tmp");
                    let entry_point = PathBuf::from("test_module.py");

                    configs.push(EncoderConfig::new(
                        key,
                        peer_keypair,
                        IpAddr::V4(Ipv4Addr::LOCALHOST),
                        encoder_port,
                        object_port,
                        project_root,
                        entry_point,
                    ));
                }
                configs
            }
        };

        // Now that we have all encoder configs, we need to update them with the collective knowledge

        // 1. Collect all encoder public keys for the committee
        let all_encoder_keys: Vec<EncoderPublicKey> = encoder_configs
            .iter()
            .map(|config| config.protocol_public_key())
            .collect();

        // 2. Create a mapping of peer keys to encoder keys
        let mut peer_to_encoder = BTreeMap::new();
        for config in &encoder_configs {
            peer_to_encoder.insert(config.peer_public_key(), config.protocol_public_key());
        }

        // 3. Create a mapping of encoder keys to (address, peer key)
        let mut encoder_to_addr_peer = BTreeMap::new();
        for config in &encoder_configs {
            encoder_to_addr_peer.insert(
                config.protocol_public_key(),
                (config.network_address.clone(), config.peer_public_key()),
            );
        }

        // 4. Create a set of allowed public keys
        let allowed_keys = encoder_configs
            .iter()
            .map(|config| config.peer_public_key().into_inner())
            .collect::<BTreeSet<_>>();

        // 5. Update each encoder config with the collective information
        for (idx, config) in encoder_configs.iter_mut().enumerate() {
            // Update context with all encoders
            config.context = EncoderConfig::create_test_context(
                &config.encoder_keypair,
                all_encoder_keys.clone(),
                idx,
            );

            // Update networking info
            config.networking_info = NetworkingInfo::new(encoder_to_addr_peer.clone());

            // Update connections info
            config.connections_info = ConnectionsInfo::new(peer_to_encoder.clone());

            // Update allowed public keys
            config.allowed_public_keys = AllowPublicKeys::new(allowed_keys.clone());
        }

        // Create nodes from the updated configs
        let nodes = encoder_configs
            .into_iter()
            .map(|config| {
                info!(
                    "Configuring encoder with name {:?}",
                    config.protocol_public_key()
                );
                (config.protocol_public_key(), Node::new(config))
            })
            .collect();

        EncoderSwarm { nodes }
    }
}
