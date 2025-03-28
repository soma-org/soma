use crate::swarm_node::Node;
use anyhow::Result;
use futures::future::try_join_all;
use node::handle::SomaNodeHandle;
use rand::rngs::OsRng;
use std::{collections::HashMap, net::SocketAddr, num::NonZeroUsize};
use tracing::info;
use types::{
    base::AuthorityName,
    config::{
        genesis_config::{AccountConfig, GenesisConfig, ValidatorGenesisConfig},
        network_config::{CommitteeConfig, ConfigBuilder, NetworkConfig},
        node_config::{FullnodeConfigBuilder, NodeConfig},
    },
};

/// A handle to an in-memory Sui Network.
#[derive(Debug)]
pub struct Swarm {
    network_config: NetworkConfig,
    nodes: HashMap<AuthorityName, Node>,
    // Save a copy of the fullnode config builder to build future fullnodes.
    fullnode_config_builder: FullnodeConfigBuilder,
}

impl Drop for Swarm {
    fn drop(&mut self) {
        self.nodes_iter_mut().for_each(|node| node.stop());
    }
}

impl Swarm {
    fn nodes_iter_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.nodes.values_mut()
    }

    /// Return a new Builder
    pub fn builder() -> SwarmBuilder {
        SwarmBuilder::new()
    }

    /// Start all nodes associated with this Swarm
    pub async fn launch(&mut self) -> Result<()> {
        try_join_all(self.nodes_iter_mut().map(|node| node.start())).await?;
        tracing::info!("Successfully launched Swarm");
        Ok(())
    }

    /// Return a reference to this Swarm's `NetworkConfig`.
    pub fn config(&self) -> &NetworkConfig {
        &self.network_config
    }

    pub fn all_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn node(&self, name: &AuthorityName) -> Option<&Node> {
        self.nodes.get(name)
    }

    /// Return an iterator over shared references of all nodes that are set up as validators.
    /// This means that they have a consensus config. This however doesn't mean this validator is
    /// currently active (i.e. it's not necessarily in the validator set at the moment).
    pub fn validator_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes
            .values()
            .filter(|node| node.config().consensus_config.is_some())
    }

    pub fn validator_node_handles(&self) -> Vec<SomaNodeHandle> {
        self.validator_nodes()
            .map(|node| node.get_node_handle().unwrap())
            .collect()
    }

    /// Returns an iterator over all currently active validators.
    pub fn active_validators(&self) -> impl Iterator<Item = &Node> {
        self.validator_nodes().filter(|node| {
            node.get_node_handle().map_or(false, |handle| {
                let state = handle.state();
                state.is_validator(&state.epoch_store_for_testing())
            })
        })
    }

    /// Return an iterator over shared references of all Fullnodes.
    pub fn fullnodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes
            .values()
            .filter(|node| node.config().consensus_config.is_none())
    }

    pub async fn spawn_new_node(&mut self, config: NodeConfig) -> SomaNodeHandle {
        let name = config.protocol_public_key();
        let node = Node::new(config);
        node.start().await.unwrap();
        let handle = node.get_node_handle().unwrap();
        self.nodes.insert(name, node);
        handle
    }

    pub fn get_fullnode_config_builder(&self) -> FullnodeConfigBuilder {
        self.fullnode_config_builder.clone()
    }
}

pub struct SwarmBuilder<R = OsRng> {
    rng: R,
    committee: CommitteeConfig,
    genesis_config: Option<GenesisConfig>,
    network_config: Option<NetworkConfig>,
    fullnode_count: usize,
    fullnode_rpc_port: Option<u16>,
    fullnode_rpc_addr: Option<SocketAddr>,
}

impl SwarmBuilder {
    pub fn new() -> Self {
        Self {
            rng: OsRng,
            committee: CommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
            genesis_config: None,
            network_config: None,
            fullnode_count: 0,
            fullnode_rpc_port: None,
            fullnode_rpc_addr: None,
        }
    }
}

impl<R> SwarmBuilder<R> {
    pub fn rng<N: rand::RngCore + rand::CryptoRng>(self, rng: N) -> SwarmBuilder<N> {
        SwarmBuilder {
            rng,
            committee: self.committee,
            genesis_config: self.genesis_config,
            network_config: self.network_config,
            fullnode_count: self.fullnode_count,
            fullnode_rpc_port: self.fullnode_rpc_port,
            fullnode_rpc_addr: self.fullnode_rpc_addr,
        }
    }

    /// Set the committee size (the number of validators in the validator set).
    ///
    /// Defaults to 1.
    pub fn committee_size(mut self, committee_size: NonZeroUsize) -> Self {
        self.committee = CommitteeConfig::Size(committee_size);
        self
    }

    pub fn with_validators(mut self, validators: Vec<ValidatorGenesisConfig>) -> Self {
        self.committee = CommitteeConfig::Validators(validators);
        self
    }

    pub fn with_genesis_config(mut self, genesis_config: GenesisConfig) -> Self {
        assert!(self.network_config.is_none() && self.genesis_config.is_none());
        self.genesis_config = Some(genesis_config);
        self
    }

    pub fn with_network_config(mut self, network_config: NetworkConfig) -> Self {
        assert!(self.network_config.is_none() && self.genesis_config.is_none());
        self.network_config = Some(network_config);
        self
    }

    pub fn with_accounts(mut self, accounts: Vec<AccountConfig>) -> Self {
        self.get_or_init_genesis_config().accounts = accounts;
        self
    }

    pub fn with_fullnode_count(mut self, fullnode_count: usize) -> Self {
        self.fullnode_count = fullnode_count;
        self
    }

    pub fn with_fullnode_rpc_port(mut self, fullnode_rpc_port: u16) -> Self {
        assert!(self.fullnode_rpc_addr.is_none());
        self.fullnode_rpc_port = Some(fullnode_rpc_port);
        self
    }

    pub fn with_fullnode_rpc_addr(mut self, fullnode_rpc_addr: SocketAddr) -> Self {
        assert!(self.fullnode_rpc_port.is_none());
        self.fullnode_rpc_addr = Some(fullnode_rpc_addr);
        self
    }

    fn get_or_init_genesis_config(&mut self) -> &mut GenesisConfig {
        if self.genesis_config.is_none() {
            assert!(self.network_config.is_none());
            self.genesis_config = Some(GenesisConfig::for_local_testing());
        }
        self.genesis_config.as_mut().unwrap()
    }
}

impl<R: rand::RngCore + rand::CryptoRng> SwarmBuilder<R> {
    /// Create the configured Swarm.
    pub fn build(self) -> Swarm {
        let network_config = self.network_config.unwrap_or_else(|| {
            let mut config_builder = ConfigBuilder::new();

            if let Some(genesis_config) = self.genesis_config {
                config_builder = config_builder.with_genesis_config(genesis_config);
            }

            config_builder
                .committee(self.committee)
                .rng(self.rng)
                .build()
        });

        let mut nodes: HashMap<_, _> = network_config
            .validator_configs()
            .iter()
            .map(|config| {
                info!(
                    "SwarmBuilder configuring validator with name {}",
                    config.protocol_public_key()
                );
                (config.protocol_public_key(), Node::new(config.to_owned()))
            })
            .collect();

        let fullnode_config_builder = FullnodeConfigBuilder::new();

        if self.fullnode_count > 0 {
            (0..self.fullnode_count).for_each(|idx| {
                let builder = fullnode_config_builder.clone();

                let config = builder.build(&mut OsRng, &network_config);
                info!(
                    "SwarmBuilder configuring full node with name {}",
                    config.protocol_public_key()
                );
                nodes.insert(config.protocol_public_key(), Node::new(config));
            });
        }
        Swarm {
            network_config,
            nodes,
            fullnode_config_builder,
        }
    }
}
