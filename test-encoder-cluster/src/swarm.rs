use anyhow::Result;
use encoder::{
    core::encoder_node::EncoderNodeHandle,
    messaging::tonic::{internal::ConnectionsInfo, NetworkingInfo},
    types::context::{Context, InnerContext},
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
use types::config::local_ip_utils;

use self::multiaddr_compat::to_network_multiaddr;
use crate::{
    config::{EncoderCommitteeConfig, EncoderConfig},
    swarm_node::Node,
};

#[derive(Debug)]
pub struct EncoderSwarm {
    nodes: HashMap<EncoderPublicKey, Node>,
    pub external_addresses: NetworkingInfo,
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
    client_keypair: Option<PeerKeyPair>,
}

impl EncoderSwarmBuilder {
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_seed([0; 32]), // TODO: change this
            committee: EncoderCommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
            client_keypair: None,
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

    pub fn with_client_keypair(mut self, keypair: PeerKeyPair) -> Self {
        self.client_keypair = Some(keypair);
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

                    // Get a unique IP for this encoder
                    let ip = local_ip_utils::get_new_ip();

                    // TODO: switch to using the find_next_available_port util!
                    // Generate network and object addresses
                    let internal_port = 8000 + (i * 3) as u16;
                    let external_port = 8001 + (i * 3) as u16;
                    let object_port = 8002 + (i * 3) as u16;
                    let probe_port = 8003 + (i * 3) as u16;

                    // Generate unique addresses with specific ports
                    let internal_network_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(
                            &ip,
                            internal_port,
                        );
                    let external_network_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(
                            &ip,
                            external_port,
                        );
                    let object_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, object_port);
                    let probe_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, probe_port);

                    let project_root = PathBuf::from("/tmp"); // Default test paths
                    let entry_point = PathBuf::from("test_module.py");

                    configs.push(EncoderConfig::new(
                        encoder_keypair,
                        peer_keypair,
                        ip.parse().unwrap(), // Parse string IP into IpAddr
                        internal_network_address,
                        external_network_address,
                        object_address,
                        probe_address,
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

                    // Get a unique IP for this encoder
                    let ip = local_ip_utils::get_new_ip();

                    let internal_port = 8000 + (i * 3) as u16;
                    let external_port = 8001 + (i * 3) as u16;
                    let object_port = 8002 + (i * 3) as u16;
                    let probe_port = 8003 + (i * 3) as u16;

                    // Generate unique addresses with specific ports
                    let internal_network_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(
                            &ip,
                            internal_port,
                        );
                    let external_network_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(
                            &ip,
                            external_port,
                        );
                    let object_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, object_port);
                    let probe_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, probe_port);

                    let project_root = PathBuf::from("/tmp");
                    let entry_point = PathBuf::from("test_module.py");

                    configs.push(EncoderConfig::new(
                        key,
                        peer_keypair,
                        ip.parse().unwrap(),
                        internal_network_address,
                        external_network_address,
                        object_address,
                        probe_address,
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

                    // For deterministic mode, use consistent port offsets based on index
                    let port_offset = 8000 + i * 10;
                    let ip = local_ip_utils::get_new_ip();

                    // Generate deterministic addresses with specific ports
                    let internal_port = 8000 + (i * 3) as u16;
                    let external_port = 8001 + (i * 3) as u16;
                    let object_port = 8002 + (i * 3) as u16;
                    let probe_port = 8003 + (i * 3) as u16;

                    // Generate unique addresses with specific ports
                    let internal_network_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(
                            &ip,
                            internal_port,
                        );
                    let external_network_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(
                            &ip,
                            external_port,
                        );
                    let object_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, object_port);
                    let probe_address =
                        local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, probe_port);

                    let project_root = PathBuf::from("/tmp");
                    let entry_point = PathBuf::from("test_module.py");

                    configs.push(EncoderConfig::new(
                        key,
                        peer_keypair,
                        ip.parse().unwrap(),
                        internal_network_address,
                        external_network_address,
                        object_address,
                        probe_address,
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
        let mut encoder_to_internal_addr_peer = BTreeMap::new();
        let mut encoder_to_external_addr_peer = BTreeMap::new();

        for config in &encoder_configs {
            // Map for internal communication (encoder-to-encoder)
            encoder_to_internal_addr_peer.insert(
                config.protocol_public_key(),
                (
                    to_network_multiaddr(&config.internal_network_address),
                    config.peer_public_key(),
                ),
            );

            // Map for external communication (client-to-encoder)
            encoder_to_external_addr_peer.insert(
                config.protocol_public_key(),
                (
                    to_network_multiaddr(&config.external_network_address),
                    config.peer_public_key(),
                ),
            );
        }

        // 4. Create a set of allowed public keys
        let mut allowed_keys = encoder_configs
            .iter()
            .map(|config| config.peer_public_key().into_inner())
            .collect::<BTreeSet<_>>();

        // Add client public key to allowed keys if available
        if let Some(client_keypair) = &self.client_keypair {
            allowed_keys.insert(client_keypair.public().into_inner());
        }

        // Collect object server information for all encoders
        let mut encoder_object_servers = HashMap::new();
        for config in &encoder_configs {
            let (key, addr) = config.get_object_server_info();
            encoder_object_servers.insert(
                key,
                (
                    config.peer_public_key(),
                    to_network_multiaddr(&config.object_address),
                ),
            );
        }

        // 5. Update each encoder config with the collective information
        for (idx, config) in encoder_configs.iter_mut().enumerate() {
            // Create a map of other encoders' object servers (excluding self)
            let mut other_object_servers = HashMap::new();
            for (key, (peer_key, addr)) in &encoder_object_servers {
                // if key != &config.protocol_public_key() {
                other_object_servers.insert(key.clone(), (peer_key.clone(), addr.clone()));
                // }
            }

            // Create context with object servers in one step
            config.context = EncoderConfig::create_test_context(
                &config.encoder_keypair,
                all_encoder_keys.clone(),
                idx,
                other_object_servers,
            );

            // Update networking info
            config.networking_info = NetworkingInfo::new(encoder_to_internal_addr_peer.clone());

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

        EncoderSwarm {
            nodes,
            external_addresses: NetworkingInfo::new(encoder_to_external_addr_peer),
        }
    }
}

// TODO: Remove this after merging Multiaddr implementations
pub mod multiaddr_compat {
    use soma_network::multiaddr as network_multiaddr;
    use types::multiaddr as local_multiaddr;

    /// Convert from types::Multiaddr to soma_network::multiaddr::Multiaddr
    pub fn to_network_multiaddr(addr: &local_multiaddr::Multiaddr) -> network_multiaddr::Multiaddr {
        // Convert through string representation
        addr.to_string().parse().expect("Valid multiaddr")
    }

    /// Convert from soma_network::multiaddr::Multiaddr to types::Multiaddr
    pub fn to_local_multiaddr(addr: &network_multiaddr::Multiaddr) -> local_multiaddr::Multiaddr {
        // Convert through string representation
        addr.to_string().parse().expect("Valid multiaddr")
    }
}
