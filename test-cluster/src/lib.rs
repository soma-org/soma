use std::{
    num::NonZeroUsize,
    time::{Duration, Instant},
};

use futures::future::join_all;
use node::handle::SomaNodeHandle;
use rand::rngs::OsRng;
use swarm::{Swarm, SwarmBuilder};
use tokio::time::timeout;
use tracing::{error, info};
use types::{
    base::{AuthorityName, ConciseableName, SomaAddress},
    committee::{CommitteeTrait, EpochId},
    config::{
        genesis_config::{
            AccountConfig, GenesisConfig, ValidatorGenesisConfig, DEFAULT_GAS_AMOUNT,
        },
        network_config::NetworkConfig,
        node_config::{FullnodeConfigBuilder, NodeConfig, ValidatorConfigBuilder},
        p2p_config::SeedPeer,
    },
    error::SomaResult,
    genesis::Genesis,
    object::ObjectRef,
    peer_id::PeerId,
    system_state::{SystemState, SystemStateTrait},
    transaction::Transaction,
};

mod container;
mod swarm;
mod swarm_node;

const NUM_VALIDATORS: usize = 4;

pub struct FullNodeHandle {
    pub soma_node: SomaNodeHandle,
}

impl FullNodeHandle {
    pub async fn new(soma_node: SomaNodeHandle) -> Self {
        Self { soma_node }
    }
}

pub struct TestCluster {
    pub swarm: Swarm,
    // pub wallet: WalletContext,
    pub fullnode_handle: FullNodeHandle,
}

impl TestCluster {
    pub fn fullnode_config_builder(&self) -> FullnodeConfigBuilder {
        self.swarm.get_fullnode_config_builder()
    }

    pub fn all_node_handles(&self) -> Vec<SomaNodeHandle> {
        self.swarm
            .all_nodes()
            .flat_map(|n| n.get_node_handle())
            .collect()
    }

    pub fn all_validator_handles(&self) -> Vec<SomaNodeHandle> {
        self.swarm
            .validator_nodes()
            .map(|n| n.get_node_handle().unwrap())
            .collect()
    }

    pub fn get_validator_pubkeys(&self) -> Vec<AuthorityName> {
        self.swarm.active_validators().map(|v| v.name()).collect()
    }

    pub fn get_genesis(&self) -> Genesis {
        self.swarm.config().genesis.clone()
    }

    pub fn stop_node(&self, name: &AuthorityName) {
        self.swarm.node(name).unwrap().stop();
    }

    pub async fn stop_all_validators(&self) {
        info!("Stopping all validators in the cluster");
        self.swarm.active_validators().for_each(|v| v.stop());
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    pub async fn start_all_validators(&self) {
        info!("Starting all validators in the cluster");
        for v in self.swarm.validator_nodes() {
            if v.is_running() {
                continue;
            }
            v.start().await.unwrap();
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }

    pub async fn start_node(&self, name: &AuthorityName) {
        let node = self.swarm.node(name).unwrap();
        if node.is_running() {
            return;
        }
        node.start().await.unwrap();
    }

    pub async fn spawn_new_validator(
        &mut self,
        genesis_config: ValidatorGenesisConfig,
    ) -> SomaNodeHandle {
        let seed_peers = self
            .swarm
            .config()
            .validator_configs
            .iter()
            .map(|config| SeedPeer {
                peer_id: Some(PeerId(
                    config.network_key_pair().public().into_inner().0.to_bytes(),
                )),
                address: config.p2p_config.external_address.clone().unwrap(),
            })
            .collect();

        let node_config = ValidatorConfigBuilder::new().build(
            genesis_config,
            self.swarm.config().genesis.clone(),
            seed_peers,
        );
        self.swarm.spawn_new_node(node_config).await
    }

    /// Convenience method to start a new fullnode in the test cluster.
    pub async fn spawn_new_fullnode(&mut self) -> FullNodeHandle {
        self.start_fullnode_from_config(
            self.fullnode_config_builder()
                .build(&mut OsRng, self.swarm.config()),
        )
        .await
    }

    pub async fn start_fullnode_from_config(&mut self, config: NodeConfig) -> FullNodeHandle {
        // let json_rpc_address = config.json_rpc_address;
        let node = self.swarm.spawn_new_node(config).await;
        FullNodeHandle::new(node).await
    }

    /// Ask 2f+1 validators to close epoch actively, and wait for the entire network to reach the next
    /// epoch. This requires waiting for both the fullnode and all validators to reach the next epoch.
    pub async fn trigger_reconfiguration(&self) {
        info!("Starting reconfiguration");
        let start = Instant::now();

        // Close epoch on 2f+1 validators.
        let cur_committee = self
            .fullnode_handle
            .soma_node
            .with(|node| node.state().clone_committee_for_testing());
        let mut cur_stake = 0;
        for node in self.swarm.active_validators() {
            node.get_node_handle()
                .unwrap()
                .with_async(|node| async {
                    node.close_epoch_for_testing().await.unwrap();
                    cur_stake += cur_committee.weight(&node.state().name);
                })
                .await;
            if cur_stake >= cur_committee.quorum_threshold() {
                break;
            }
        }
        info!("close_epoch complete after {:?}", start.elapsed());

        self.wait_for_epoch(Some(cur_committee.epoch + 1)).await;
        self.wait_for_epoch_all_nodes(cur_committee.epoch + 1).await;

        info!("reconfiguration complete after {:?}", start.elapsed());
    }

    /// To detect whether the network has reached such state, we use the fullnode as the
    /// source of truth, since a fullnode only does epoch transition when the network has
    /// done so.
    /// If target_epoch is specified, wait until the cluster reaches that epoch.
    /// If target_epoch is None, wait until the cluster reaches the next epoch.
    /// Note that this function does not guarantee that every node is at the target epoch.
    pub async fn wait_for_epoch(&self, target_epoch: Option<EpochId>) -> SystemState {
        self.wait_for_epoch_with_timeout(target_epoch, Duration::from_secs(120))
            .await
    }

    pub async fn wait_for_epoch_on_node(
        &self,
        handle: &SomaNodeHandle,
        target_epoch: Option<EpochId>,
        timeout_dur: Duration,
    ) -> SystemState {
        let mut epoch_rx = handle.with(|node| node.subscribe_to_epoch_change());

        let mut state = None;
        timeout(timeout_dur, async {
            let epoch = handle.with(|node| node.state().epoch_store_for_testing().epoch());
            if Some(epoch) == target_epoch {
                return handle.with(|node| node.state().get_system_state_object_for_testing());
            }
            while let Ok(system_state) = epoch_rx.recv().await {
                info!("received epoch {}", system_state.epoch());
                state = Some(system_state.clone());
                match target_epoch {
                    Some(target_epoch) if system_state.epoch() >= target_epoch => {
                        return system_state;
                    }
                    None => {
                        return system_state;
                    }
                    _ => (),
                }
            }
            unreachable!("Broken reconfig channel");
        })
        .await
        .unwrap_or_else(|_| {
            error!("Timed out waiting for cluster to reach epoch {target_epoch:?}");
            if let Some(state) = state {
                panic!("Timed out waiting for cluster to reach epoch {target_epoch:?}. Current epoch: {}", state.epoch());
            }
            panic!("Timed out waiting for cluster to target epoch {target_epoch:?}")
        })
    }

    pub async fn wait_for_epoch_with_timeout(
        &self,
        target_epoch: Option<EpochId>,
        timeout_dur: Duration,
    ) -> SystemState {
        self.wait_for_epoch_on_node(&self.fullnode_handle.soma_node, target_epoch, timeout_dur)
            .await
    }

    pub async fn wait_for_epoch_all_nodes(&self, target_epoch: EpochId) {
        let handles: Vec<_> = self
            .swarm
            .all_nodes()
            .map(|node| node.get_node_handle().unwrap())
            .collect();
        let tasks: Vec<_> = handles
            .iter()
            .map(|handle| {
                handle.with_async(|node| async {
                    let mut retries = 0;
                    loop {
                        let epoch = node.state().epoch_store_for_testing().epoch();
                        if epoch == target_epoch {
                            if let Some(agg) = node.clone_authority_aggregator() {
                                // This is a fullnode, we need to wait for its auth aggregator to reconfigure as well.
                                if agg.committee.epoch() == target_epoch {
                                    break;
                                }
                            } else {
                                // This is a validator, we don't need to check the auth aggregator.
                                break;
                            }
                        }
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        retries += 1;
                        if retries % 5 == 0 {
                            tracing::warn!(validator=?node.state().name.concise(), "Waiting for {:?} seconds to reach epoch {:?}. Currently at epoch {:?}", retries, target_epoch, epoch);
                        }
                    }
                })
            })
            .collect();

        timeout(Duration::from_secs(40), join_all(tasks))
            .await
            .expect("timed out waiting for reconfiguration to complete");
    }

    pub async fn execute_transaction(&self, tx: Transaction) -> SomaResult {
        self.fullnode_handle
            .soma_node
            .with_async(|node| async { node.execute_transaction(tx).await })
            .await
    }

    pub async fn get_gas_objects_owned_by_address(
        &self,
        address: SomaAddress,
        limit: Option<usize>,
    ) -> SomaResult<Vec<ObjectRef>> {
        self.fullnode_handle
            .soma_node
            .with_async(|node| async {
                node.get_gas_objects_owned_by_address(address, limit).await
            })
            .await
    }
}

pub struct TestClusterBuilder {
    num_validators: Option<usize>,
    genesis_config: Option<GenesisConfig>,
    network_config: Option<NetworkConfig>,
}

impl TestClusterBuilder {
    pub fn new() -> Self {
        TestClusterBuilder {
            num_validators: None,
            genesis_config: None,
            network_config: None,
        }
    }

    pub fn with_num_validators(mut self, num: usize) -> Self {
        self.num_validators = Some(num);
        self
    }

    pub fn set_genesis_config(mut self, genesis_config: GenesisConfig) -> Self {
        assert!(self.genesis_config.is_none() && self.network_config.is_none());
        self.genesis_config = Some(genesis_config);
        self
    }

    pub fn set_network_config(mut self, network_config: NetworkConfig) -> Self {
        assert!(self.genesis_config.is_none() && self.network_config.is_none());
        self.network_config = Some(network_config);
        self
    }

    pub fn with_epoch_duration_ms(mut self, epoch_duration_ms: u64) -> Self {
        self.get_or_init_genesis_config()
            .parameters
            .epoch_duration_ms = epoch_duration_ms;
        self
    }

    pub async fn build(mut self) -> TestCluster {
        let swarm = self.start_swarm().await.unwrap();
        let fullnode = swarm.fullnodes().next().unwrap();
        let fullnode_handle = fullnode.get_node_handle().unwrap();

        TestCluster {
            fullnode_handle: FullNodeHandle {
                soma_node: fullnode_handle,
            },
            swarm,
        }
    }

    async fn start_swarm(&mut self) -> Result<Swarm, anyhow::Error> {
        let mut builder: SwarmBuilder = Swarm::builder()
            .committee_size(
                NonZeroUsize::new(self.num_validators.unwrap_or(NUM_VALIDATORS)).unwrap(),
            )
            .with_fullnode_count(1);

        if let Some(genesis_config) = self.genesis_config.take() {
            builder = builder.with_genesis_config(genesis_config);
        }

        if let Some(network_config) = self.network_config.take() {
            builder = builder.with_network_config(network_config);
        }

        let mut swarm = builder.build();
        swarm.launch().await?;
        Ok(swarm)
    }

    fn get_or_init_genesis_config(&mut self) -> &mut GenesisConfig {
        if self.genesis_config.is_none() {
            self.genesis_config = Some(GenesisConfig::for_local_testing());
        }
        self.genesis_config.as_mut().unwrap()
    }

    pub fn with_validator_candidates(
        mut self,
        addresses: impl IntoIterator<Item = SomaAddress>,
    ) -> Self {
        self.get_or_init_genesis_config()
            .accounts
            .extend(addresses.into_iter().map(|address| AccountConfig {
                address: Some(address),
                gas_amounts: vec![DEFAULT_GAS_AMOUNT, DEFAULT_GAS_AMOUNT],
            }));
        self
    }
}

impl Default for TestClusterBuilder {
    fn default() -> Self {
        Self::new()
    }
}
