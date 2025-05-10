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
use types::{committee::Committee, config::local_ip_utils};

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

                // TODO: THIS IS TEMPORARY TILL YOU CONNECT THE ENCODERS WITH ACTUAL VALIDATORS/FULLNODES
                // Create default genesis committee that all encoders will share
                let (genesis_committee, _) = Committee::new_simple_test_committee_of_size(4);

                // Create validator rpc all encoders will share
                let validator_ip = local_ip_utils::get_new_ip();
                let validator_rpc_address =
                    local_ip_utils::new_tcp_address_for_testing(&validator_ip);

                for i in 0..size.get() {
                    // Generate keypairs
                    let encoder_keypair = EncoderKeyPair::generate(&mut self.rng);
                    let peer_keypair = PeerKeyPair::generate(&mut self.rng);

                    // Get a unique IP for this encoder
                    let ip = local_ip_utils::get_new_ip();

                    // Generate unique addresses with specific ports
                    let internal_network_address = local_ip_utils::new_tcp_address_for_testing(&ip);
                    let external_network_address = local_ip_utils::new_tcp_address_for_testing(&ip);
                    let object_address = local_ip_utils::new_tcp_address_for_testing(&ip);
                    let probe_address = local_ip_utils::new_tcp_address_for_testing(&ip);

                    let project_root = PathBuf::from("/tmp"); // Default test paths
                    let entry_point = PathBuf::from("test_module.py");

                    configs.push(EncoderConfig::new(
                        encoder_keypair,
                        peer_keypair,
                        internal_network_address,
                        external_network_address,
                        object_address,
                        probe_address,
                        project_root,
                        entry_point,
                        validator_rpc_address.clone(),
                        genesis_committee.clone(),
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

                // TODO: THIS IS TEMPORARY TILL YOU CONNECT THE ENCODERS WITH ACTUAL VALIDATORS/FULLNODES
                // Create default genesis committee that all encoders will share
                let (genesis_committee, _) = Committee::new_simple_test_committee_of_size(4);

                // Create validator rpc all encoders will share
                let validator_ip = local_ip_utils::get_new_ip();
                let validator_rpc_address =
                    local_ip_utils::new_tcp_address_for_testing(&validator_ip);

                for (i, key) in keys.into_iter().enumerate() {
                    // Generate peer keypair
                    let peer_keypair = PeerKeyPair::generate(&mut self.rng);

                    // Get a unique IP for this encoder
                    let ip = local_ip_utils::get_new_ip();

                    let internal_network_address = local_ip_utils::new_tcp_address_for_testing(&ip);
                    let external_network_address = local_ip_utils::new_tcp_address_for_testing(&ip);
                    let object_address = local_ip_utils::new_tcp_address_for_testing(&ip);
                    let probe_address = local_ip_utils::new_tcp_address_for_testing(&ip);

                    let project_root = PathBuf::from("/tmp");
                    let entry_point = PathBuf::from("test_module.py");

                    configs.push(EncoderConfig::new(
                        key,
                        peer_keypair,
                        internal_network_address,
                        external_network_address,
                        object_address,
                        probe_address,
                        project_root,
                        entry_point,
                        validator_rpc_address.clone(),
                        genesis_committee.clone(),
                    ));
                }
                configs
            }
        };

        // Create nodes from the updated configs
        let nodes = encoder_configs
            .clone()
            .into_iter()
            .map(|config| {
                info!(
                    "Configuring encoder with name {:?}",
                    config.protocol_public_key()
                );
                (config.protocol_public_key(), Node::new(config))
            })
            .collect();

        let external_addresses = encoder_configs
            .iter()
            .map(|config| {
                (
                    config.protocol_public_key(),
                    (
                        to_network_multiaddr(&config.external_network_address),
                        config.peer_public_key(),
                    ),
                )
            })
            .collect();

        EncoderSwarm {
            nodes,
            external_addresses: NetworkingInfo::new(external_addresses),
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
