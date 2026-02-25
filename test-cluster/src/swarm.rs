// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

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
        genesis_config::{AccountConfig, GenesisConfig, ValidatorGenesisConfig},
        network_config::{
            CommitteeConfig, ConfigBuilder, NetworkConfig, ProtocolVersionsConfig,
            SupportedProtocolVersionsCallback,
        },
        node_config::{FullnodeConfigBuilder, NodeConfig},
        p2p_config::SeedPeer,
    },
    multiaddr::Multiaddr,
    peer_id::PeerId,
    supported_protocol_versions::{ProtocolVersion, SupportedProtocolVersions},
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

/// A handle to an in-memory SOMA Network.
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
        self.nodes.values().filter(|node| node.config().consensus_config.is_some())
    }

    pub fn validator_node_handles(&self) -> Vec<SomaNodeHandle> {
        self.validator_nodes().map(|node| node.get_node_handle().unwrap()).collect()
    }

    /// Returns an iterator over all currently active validators.
    pub fn active_validators(&self) -> impl Iterator<Item = &Node> {
        self.validator_nodes().filter(|node| {
            node.get_node_handle().is_some_and(|handle| {
                let state = handle.state();
                state.is_validator(&state.epoch_store_for_testing())
            })
        })
    }

    /// Return an iterator over shared references of all Fullnodes.
    pub fn fullnodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values().filter(|node| node.config().consensus_config.is_none())
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
    genesis_config: Option<GenesisConfig>,
    network_config: Option<NetworkConfig>,
    fullnode_count: usize,
    fullnode_rpc_port: Option<u16>,
    fullnode_rpc_addr: Option<SocketAddr>,
    fullnode_rpc_config: Option<types::config::rpc_config::RpcConfig>,
    supported_protocol_versions_config: ProtocolVersionsConfig,
    data_ingestion_dir: Option<PathBuf>,
    fullnode_run_with_range: Option<types::config::node_config::RunWithRange>,
    scoring_url: Option<String>,
}

impl Default for SwarmBuilder {
    fn default() -> Self {
        Self {
            rng: OsRng,
            dir: None,
            committee: CommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
            genesis_config: None,
            network_config: None,
            fullnode_count: 0,
            fullnode_rpc_port: None,
            fullnode_rpc_addr: None,
            fullnode_rpc_config: None,
            supported_protocol_versions_config: ProtocolVersionsConfig::Default,
            data_ingestion_dir: None,
            fullnode_run_with_range: None,
            scoring_url: None,
        }
    }
}

impl SwarmBuilder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<R> SwarmBuilder<R> {
    pub fn rng<N: rand::RngCore + rand::CryptoRng>(self, rng: N) -> SwarmBuilder<N> {
        SwarmBuilder {
            rng,
            dir: self.dir,
            committee: self.committee,
            genesis_config: self.genesis_config,
            network_config: self.network_config,
            fullnode_count: self.fullnode_count,
            fullnode_rpc_port: self.fullnode_rpc_port,
            fullnode_rpc_addr: self.fullnode_rpc_addr,
            fullnode_rpc_config: self.fullnode_rpc_config,
            supported_protocol_versions_config: self.supported_protocol_versions_config,
            data_ingestion_dir: self.data_ingestion_dir,
            fullnode_run_with_range: self.fullnode_run_with_range,
            scoring_url: self.scoring_url,
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

    pub fn with_protocol_version(mut self, v: ProtocolVersion) -> Self {
        self.get_or_init_genesis_config().parameters.protocol_version = v;
        self
    }

    pub fn with_supported_protocol_versions(mut self, c: SupportedProtocolVersions) -> Self {
        self.supported_protocol_versions_config = ProtocolVersionsConfig::Global(c);
        self
    }

    pub fn with_supported_protocol_version_callback(
        mut self,
        func: SupportedProtocolVersionsCallback,
    ) -> Self {
        self.supported_protocol_versions_config = ProtocolVersionsConfig::PerValidator(func);
        self
    }

    pub fn with_supported_protocol_versions_config(mut self, c: ProtocolVersionsConfig) -> Self {
        self.supported_protocol_versions_config = c;
        self
    }

    pub fn with_fullnode_rpc_config(
        mut self,
        fullnode_rpc_config: types::config::rpc_config::RpcConfig,
    ) -> Self {
        self.fullnode_rpc_config = Some(fullnode_rpc_config);
        self
    }

    pub fn with_epoch_duration_ms(mut self, epoch_duration_ms: u64) -> Self {
        self.get_or_init_genesis_config().parameters.epoch_duration_ms = epoch_duration_ms;
        self
    }

    pub fn with_data_ingestion_dir(mut self, path: PathBuf) -> Self {
        self.data_ingestion_dir = Some(path);
        self
    }

    pub fn with_fullnode_run_with_range(
        mut self,
        run_with_range: types::config::node_config::RunWithRange,
    ) -> Self {
        self.fullnode_run_with_range = Some(run_with_range);
        self
    }

    pub fn with_scoring_url(mut self, url: String) -> Self {
        self.scoring_url = Some(url);
        self
    }
}

// TODO: modify this build to make use of fullnode configs and data ingestion urls
impl<R: rand::RngCore + rand::CryptoRng> SwarmBuilder<R> {
    /// Create the configured Swarm.
    pub fn build(self) -> Swarm {
        let dir = if let Some(dir) = self.dir {
            SwarmDirectory::Persistent(dir)
        } else {
            SwarmDirectory::new_temporary()
        };

        let mut network_config = self.network_config.unwrap_or_else(|| {
            let mut config_builder = ConfigBuilder::new(dir.as_ref());

            if let Some(genesis_config) = self.genesis_config {
                config_builder = config_builder.with_genesis_config(genesis_config);
            }

            config_builder
                .committee(self.committee)
                .with_supported_protocol_versions_config(
                    self.supported_protocol_versions_config.clone(),
                )
                .rng(self.rng)
                .build()
        });

        // Apply scoring_url to validator configs if set
        if let Some(ref scoring_url) = self.scoring_url {
            for config in &mut network_config.validator_configs {
                config.scoring_url = Some(scoring_url.clone());
            }
        }

        // Create validator nodes
        let mut nodes: HashMap<_, _> = network_config
            .validator_configs()
            .iter()
            .map(|config| {
                info!("SwarmBuilder configuring validator {}", config.protocol_public_key());
                (config.protocol_public_key(), Node::new(config.to_owned()))
            })
            .collect();

        // Create fullnode nodes using FullnodeConfigBuilder
        if self.fullnode_count > 0 {
            // Extract seed peers from validator configs
            let seed_peers: Vec<SeedPeer> = network_config
                .validator_configs()
                .iter()
                .filter_map(|config| {
                    let p2p_address = config.p2p_config.external_address.clone()?;
                    Some(SeedPeer {
                        peer_id: Some(PeerId(
                            config.network_key_pair().public().into_inner().0.to_bytes(),
                        )),
                        address: p2p_address,
                    })
                })
                .collect();

            let genesis = network_config.genesis.clone();

            for i in 0..self.fullnode_count {
                let mut builder =
                    FullnodeConfigBuilder::new().with_config_directory(dir.as_ref().to_path_buf());

                if let Some(rpc_addr) = self.fullnode_rpc_addr {
                    builder = builder.with_rpc_addr(rpc_addr);
                } else if let Some(rpc_port) = self.fullnode_rpc_port {
                    builder = builder.with_rpc_port(rpc_port);
                }

                if let Some(ref rpc_config) = self.fullnode_rpc_config {
                    builder = builder.with_rpc_config(rpc_config.clone());
                }

                if let Some(run_with_range) = self.fullnode_run_with_range {
                    builder = builder.with_run_with_range(run_with_range);
                }

                let fullnode_config = builder.build(genesis.clone(), seed_peers.clone());

                info!(
                    "SwarmBuilder configuring fullnode {} ({})",
                    i,
                    fullnode_config.protocol_public_key()
                );

                nodes.insert(fullnode_config.protocol_public_key(), Node::new(fullnode_config));
            }
        }

        Swarm { network_config, nodes, dir }
    }
}
