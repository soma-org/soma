use crate::swarm_node::Node;
use anyhow::Result;
use futures::future::try_join_all;
use node::handle::SomaNodeHandle;
use rand::rngs::OsRng;
use std::{
    collections::HashMap,
    net::SocketAddr,
    num::NonZeroUsize,
    path::{Path, PathBuf},
};
use tempfile::TempDir;
use tracing::info;
use types::{
    base::AuthorityName,
    config::{
        encoder_config::{EncoderCommitteeConfig, EncoderConfig, EncoderGenesisConfig},
        genesis_config::{AccountConfig, GenesisConfig, ValidatorGenesisConfig},
        network_config::{CommitteeConfig, ConfigBuilder, NetworkConfig},
        node_config::NodeConfig,
    },
    multiaddr::Multiaddr,
};

#[derive(Debug)]
enum SwarmDirectory {
    Persistent(PathBuf),
    Temporary(TempDir),
}

impl SwarmDirectory {
    fn new_temporary() -> Self {
        SwarmDirectory::Temporary(TempDir::new().unwrap())
    }
}

impl std::ops::Deref for SwarmDirectory {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        match self {
            SwarmDirectory::Persistent(dir) => dir.deref(),
            SwarmDirectory::Temporary(dir) => dir.path(),
        }
    }
}

impl AsRef<Path> for SwarmDirectory {
    fn as_ref(&self) -> &Path {
        match self {
            SwarmDirectory::Persistent(dir) => dir.as_ref(),
            SwarmDirectory::Temporary(dir) => dir.as_ref(),
        }
    }
}

/// A handle to an in-memory Soma Network.
#[derive(Debug)]
pub struct Swarm {
    dir: SwarmDirectory,
    network_config: NetworkConfig,
    nodes: HashMap<AuthorityName, Node>,
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

    pub fn encoder_configs(&self) -> impl Iterator<Item = &EncoderConfig> {
        self.network_config.encoder_configs.iter()
    }

    /// Return a reference to an encoder config by index.
    pub fn encoder_config(&self, index: usize) -> Option<&EncoderConfig> {
        self.network_config.encoder_configs.get(index)
    }

    pub fn all_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn node(&self, name: &AuthorityName) -> Option<&Node> {
        self.nodes.get(name)
    }

    /// Return the path to the directory where this Swarm's on-disk data is kept.
    pub fn dir(&self) -> &Path {
        self.dir.as_ref()
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
}

pub struct SwarmBuilder<R = OsRng> {
    rng: R,
    dir: Option<PathBuf>,
    committee: CommitteeConfig,
    encoder_committee: EncoderCommitteeConfig,
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
            dir: None,
            committee: CommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
            encoder_committee: EncoderCommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
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
            dir: self.dir,
            committee: self.committee,
            encoder_committee: self.encoder_committee,
            genesis_config: self.genesis_config,
            network_config: self.network_config,
            fullnode_count: self.fullnode_count,
            fullnode_rpc_port: self.fullnode_rpc_port,
            fullnode_rpc_addr: self.fullnode_rpc_addr,
        }
    }

    /// Set the directory that should be used by the Swarm for any on-disk data.
    ///
    /// If a directory is provided, it will not be cleaned up when the Swarm is dropped.
    ///
    /// Defaults to using a temporary directory that will be cleaned up when the Swarm is dropped.
    pub fn dir<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.dir = Some(dir.into());
        self
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

    pub fn encoder_committee_size(mut self, committee_size: NonZeroUsize) -> Self {
        self.encoder_committee = EncoderCommitteeConfig::Size(committee_size);
        self
    }

    pub fn with_encoders(mut self, encoders: Vec<EncoderGenesisConfig>) -> Self {
        self.encoder_committee = EncoderCommitteeConfig::Encoders(encoders);
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
        let dir = if let Some(dir) = self.dir {
            SwarmDirectory::Persistent(dir)
        } else {
            SwarmDirectory::new_temporary()
        };

        let mut network_config = self.network_config.unwrap_or_else(|| {
            let mut config_builder = ConfigBuilder::new();

            if let Some(genesis_config) = self.genesis_config {
                config_builder = config_builder.with_genesis_config(genesis_config);
            }

            // Automatically add networking validators if fullnode_count > 0
            let committee = if self.fullnode_count > 0 {
                match self.committee {
                    CommitteeConfig::Size(consensus_size) => CommitteeConfig::Mixed {
                        consensus_count: consensus_size,
                        networking_count: NonZeroUsize::new(self.fullnode_count).unwrap(),
                    },
                    other => other, // Keep as-is for other variants
                }
            } else {
                self.committee
            };

            config_builder
                .committee(committee)
                .encoder_committee(self.encoder_committee)
                .rng(self.rng)
                .build()
        });

        let networking_validator_addresses: Vec<Multiaddr> = network_config
            .validator_configs()
            .iter()
            .filter(|c| c.consensus_config.is_none())
            .map(|c| c.encoder_validator_address.clone())
            .collect();

        // Now update encoder configs without holding any borrows to validator_configs
        if !networking_validator_addresses.is_empty() && !network_config.encoder_configs.is_empty()
        {
            for (i, encoder_config) in network_config.encoder_configs.iter_mut().enumerate() {
                let validator_index = i % networking_validator_addresses.len();
                encoder_config.validator_sync_address =
                    networking_validator_addresses[validator_index].clone();

                info!("Assigned encoder {} to networking validator", i);
            }
        } else if !network_config.encoder_configs.is_empty() {
            // Fallback: get first validator address
            if let Some(first_validator_address) = network_config
                .validator_configs()
                .first()
                .map(|c| c.encoder_validator_address.clone())
            {
                for encoder_config in network_config.encoder_configs.iter_mut() {
                    encoder_config.validator_sync_address = first_validator_address.clone();
                }
            }
        }

        // Create all validator nodes
        let mut nodes: HashMap<_, _> = network_config
            .validator_configs()
            .iter()
            .map(|config| {
                let node_type = if config.consensus_config.is_some() {
                    "consensus validator"
                } else {
                    "networking validator"
                };
                info!(
                    "SwarmBuilder configuring {} {}",
                    node_type,
                    config.protocol_public_key()
                );
                (config.protocol_public_key(), Node::new(config.to_owned()))
            })
            .collect();

        Swarm {
            network_config,
            nodes,
            dir,
        }
    }
}
