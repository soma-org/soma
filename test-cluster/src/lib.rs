use std::{
    net::SocketAddr,
    num::NonZeroUsize,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use bytes::Bytes;
use fastcrypto::hash::HashFunction as _;
use futures::future::join_all;
use node::handle::SomaNodeHandle;
use object_store::{ObjectStore as _, memory::InMemory};
use rand::rngs::OsRng;
use rpc::api::client::{TransactionExecutionResponse, TransactionExecutionResponseWithCheckpoint};
use sdk::{
    SomaClient, SomaClientBuilder,
    client_config::{SomaClientConfig, SomaEnv},
    wallet_context::WalletContext,
};
use soma_keys::keystore::{AccountKeystore, FileBasedKeystore, Keystore};
use swarm::{Swarm, SwarmBuilder};
use tokio::time::timeout;
use tracing::{error, info};
use types::{
    base::{AuthorityName, ConciseableName, SomaAddress},
    checksum::Checksum,
    committee::{CommitteeTrait, EpochId},
    config::{
        Config, PersistedConfig, SOMA_CLIENT_CONFIG, SOMA_KEYSTORE_FILENAME, SOMA_NETWORK_CONFIG,
        genesis_config::{
            AccountConfig, DEFAULT_GAS_AMOUNT, GenesisConfig, GenesisModelConfig,
            ValidatorGenesisConfig,
        },
        local_ip_utils,
        network_config::{
            NetworkConfig, ProtocolVersionsConfig, SupportedProtocolVersionsCallback,
        },
        node_config::{NodeConfig, RunWithRange, ValidatorConfigBuilder, get_key_path},
        p2p_config::SeedPeer,
    },
    crypto::{DefaultHash, NetworkKeyPair, SomaKeyPair},
    effects::TransactionEffects,
    error::SomaResult,
    genesis::Genesis,
    object::ObjectRef,
    peer_id::PeerId,
    supported_protocol_versions::{ProtocolVersion, SupportedProtocolVersions},
    system_state::{SystemState, SystemStateTrait, epoch_start::EpochStartSystemStateTrait as _},
    transaction::{Transaction, TransactionData},
};
use url::Url;

#[cfg(msim)]
#[path = "./container-sim.rs"]
mod container;

#[cfg(not(msim))]
#[path = "./container.rs"]
mod container;

pub mod swarm;
mod swarm_node;

const NUM_VALIDATORS: usize = 4;

/// Mock scoring gRPC service for e2e tests.
/// Returns deterministic results (winner=0, all-zero scores) without real ML inference.
struct MockScoringService;

#[scoring::tonic::async_trait]
impl scoring::tonic_gen::scoring_server::Scoring for MockScoringService {
    async fn score(
        &self,
        request: scoring::tonic::Request<scoring::types::ScoreRequest>,
    ) -> Result<scoring::tonic::Response<scoring::types::ScoreResponse>, scoring::tonic::Status>
    {
        let req = request.into_inner();
        let num_models = req.model_manifests.len().max(1);
        Ok(scoring::tonic::Response::new(scoring::types::ScoreResponse {
            winner: 0,
            loss_score: vec![0.0; num_models],
            embedding: vec![0.0; req.target_embedding.len()],
            distance: vec![0.0; num_models],
        }))
    }

    async fn health(
        &self,
        _request: scoring::tonic::Request<scoring::types::HealthRequest>,
    ) -> Result<scoring::tonic::Response<scoring::types::HealthResponse>, scoring::tonic::Status>
    {
        Ok(scoring::tonic::Response::new(scoring::types::HealthResponse { ok: true }))
    }
}

pub struct FullNodeHandle {
    pub soma_node: SomaNodeHandle,
    pub soma_client: SomaClient,
    pub rpc_url: String,
}

impl FullNodeHandle {
    pub async fn new(soma_node: SomaNodeHandle, rpc_address: SocketAddr) -> Self {
        let rpc_url = format!("http://{}", rpc_address);
        let soma_client = SomaClientBuilder::default().build(&rpc_url).await.unwrap();

        Self { soma_node, soma_client, rpc_url }
    }
}

pub struct TestCluster {
    pub swarm: Swarm,
    pub wallet: WalletContext,
    pub fullnode_handle: FullNodeHandle,
}

impl TestCluster {
    pub fn wallet(&mut self) -> &WalletContext {
        &self.wallet
    }

    pub fn wallet_mut(&mut self) -> &mut WalletContext {
        &mut self.wallet
    }

    pub fn get_addresses(&self) -> Vec<SomaAddress> {
        self.wallet.get_addresses()
    }

    pub fn all_node_handles(&self) -> Vec<SomaNodeHandle> {
        self.swarm.all_nodes().flat_map(|n| n.get_node_handle()).collect()
    }

    pub fn all_validator_handles(&self) -> Vec<SomaNodeHandle> {
        self.swarm.validator_nodes().map(|n| n.get_node_handle().unwrap()).collect()
    }

    pub fn get_validator_pubkeys(&self) -> Vec<AuthorityName> {
        self.swarm.active_validators().map(|v| v.name()).collect()
    }

    pub fn get_genesis(&self) -> types::config::node_config::Genesis {
        types::config::node_config::Genesis::new(self.swarm.config().genesis.clone())
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
                peer_id: Some(PeerId(config.network_key_pair().public().into_inner().0.to_bytes())),
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

    pub async fn start_fullnode_from_config(&mut self, config: NodeConfig) -> FullNodeHandle {
        let rpc_address = config.rpc_address;
        let node = self.swarm.spawn_new_node(config).await;
        FullNodeHandle::new(node, rpc_address).await
    }

    pub async fn wait_for_run_with_range_shutdown_signal(&self) -> Option<RunWithRange> {
        self.wait_for_run_with_range_shutdown_signal_with_timeout(Duration::from_secs(60)).await
    }

    pub async fn wait_for_run_with_range_shutdown_signal_with_timeout(
        &self,
        timeout_dur: Duration,
    ) -> Option<RunWithRange> {
        let mut shutdown_channel_rx =
            self.fullnode_handle.soma_node.with(|node| node.subscribe_to_shutdown_channel());

        timeout(timeout_dur, async move {
            tokio::select! {
                msg = shutdown_channel_rx.recv() =>
                {
                    match msg {
                        Ok(Some(run_with_range)) => Some(run_with_range),
                        Ok(None) => None,
                        Err(e) => {
                            error!("failed recv from soma-node shutdown channel: {}", e);
                            None
                        },
                    }
                },
            }
        })
        .await
        .expect("Timed out waiting for cluster to hit target epoch and recv shutdown signal from sui-node")
    }

    pub async fn wait_for_protocol_version(
        &self,
        target_protocol_version: ProtocolVersion,
    ) -> SystemState {
        self.wait_for_protocol_version_with_timeout(
            target_protocol_version,
            Duration::from_secs(60),
        )
        .await
    }

    pub async fn wait_for_protocol_version_with_timeout(
        &self,
        target_protocol_version: ProtocolVersion,
        timeout_dur: Duration,
    ) -> SystemState {
        timeout(timeout_dur, async move {
            loop {
                let system_state = self.wait_for_epoch(None).await;
                if system_state.protocol_version() >= target_protocol_version.as_u64() {
                    return system_state;
                }
            }
        })
        .await
        .expect("Timed out waiting for cluster to target protocol version")
    }

    /// Ask 2f+1 validators to close epoch actively, and wait for the entire network to reach the next
    /// epoch. This requires waiting for both the fullnode and all validators to reach the next epoch.
    pub async fn trigger_reconfiguration(&self) {
        info!("Starting reconfiguration");
        let start = Instant::now();

        // Close epoch on 2f+1 validators.
        let cur_committee =
            self.fullnode_handle.soma_node.with(|node| node.state().clone_committee_for_testing());
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
        self.wait_for_epoch_with_timeout(target_epoch, Duration::from_secs(120)).await
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
                return handle.with(|node| node.state().get_system_state_object_for_testing().unwrap());
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
        let handles: Vec<_> =
            self.swarm.all_nodes().map(|node| node.get_node_handle().unwrap()).collect();
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

    /// Upgrade the network protocol version, by restarting every validator with a new
    /// supported versions.
    /// Note that we don't restart the fullnode here, and it is assumed that the fulnode supports
    /// the entire version range.
    pub async fn update_validator_supported_versions(
        &self,
        new_supported_versions: SupportedProtocolVersions,
    ) {
        for authority in self.get_validator_pubkeys() {
            self.stop_node(&authority);
            tokio::time::sleep(Duration::from_millis(1000)).await;
            self.swarm.node(&authority).unwrap().config().supported_protocol_versions =
                Some(new_supported_versions);
            self.start_node(&authority).await;
            info!("Restarted validator {}", authority);
        }
    }

    /// Wait for all nodes in the network to upgrade to `protocol_version`.
    pub async fn wait_for_all_nodes_upgrade_to(&self, protocol_version: u64) {
        for h in self.all_node_handles() {
            h.with_async(|node| async {
                while node
                    .state()
                    .epoch_store_for_testing()
                    .epoch_start_state()
                    .protocol_version()
                    .as_u64()
                    != protocol_version
                {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            })
            .await;
        }
    }

    /// Return the highest observed protocol version in the test cluster.
    pub fn highest_protocol_version(&self) -> ProtocolVersion {
        self.all_node_handles()
            .into_iter()
            .map(|h| {
                h.with(|node| {
                    node.state().epoch_store_for_testing().epoch_start_state().protocol_version()
                })
            })
            .max()
            .expect("at least one node must be up to get highest protocol version")
    }

    pub async fn sign_transaction(&self, tx_data: &TransactionData) -> Transaction {
        self.wallet.sign_transaction(tx_data).await
    }

    pub async fn sign_and_execute_transaction(
        &self,
        tx_data: &TransactionData,
    ) -> TransactionExecutionResponseWithCheckpoint {
        let tx = self.wallet.sign_transaction(tx_data).await;
        self.execute_transaction(tx).await
    }

    /// Execute a transaction on the network and wait for it to be executed on the rpc fullnode.
    /// Also expects the effects status to be ExecutionStatus::Success.
    /// This function is recommended for transaction execution since it most resembles the
    /// production path.
    pub async fn execute_transaction(
        &self,
        tx: Transaction,
    ) -> TransactionExecutionResponseWithCheckpoint {
        self.wallet
            .execute_transaction_and_wait_for_indexing(tx)
            .await
            .expect("Transaction must succeed")
    }
}

pub struct TestClusterBuilder {
    num_validators: Option<usize>,
    genesis_config: Option<GenesisConfig>,
    network_config: Option<NetworkConfig>,
    validators: Option<Vec<ValidatorGenesisConfig>>,
    validator_supported_protocol_versions_config: ProtocolVersionsConfig,
    fullnode_run_with_range: Option<RunWithRange>,
}

impl TestClusterBuilder {
    pub fn new() -> Self {
        TestClusterBuilder {
            num_validators: None,
            genesis_config: None,
            network_config: None,
            validators: None,
            validator_supported_protocol_versions_config: ProtocolVersionsConfig::Default,
            fullnode_run_with_range: None,
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
        self.get_or_init_genesis_config().parameters.epoch_duration_ms = epoch_duration_ms;
        self
    }

    /// Provide validator genesis configs, overrides the `num_validators` setting.
    pub fn with_validators(mut self, validators: Vec<ValidatorGenesisConfig>) -> Self {
        self.validators = Some(validators);
        self
    }

    pub fn with_accounts(mut self, accounts: Vec<AccountConfig>) -> Self {
        self.get_or_init_genesis_config().accounts = accounts;
        self
    }

    pub fn with_supported_protocol_versions(mut self, c: SupportedProtocolVersions) -> Self {
        self.validator_supported_protocol_versions_config = ProtocolVersionsConfig::Global(c);
        self
    }

    pub fn with_protocol_version(mut self, v: ProtocolVersion) -> Self {
        self.get_or_init_genesis_config().parameters.protocol_version = v;
        self
    }

    pub fn with_supported_protocol_version_callback(
        mut self,
        func: SupportedProtocolVersionsCallback,
    ) -> Self {
        self.validator_supported_protocol_versions_config =
            ProtocolVersionsConfig::PerValidator(func);
        self
    }

    pub fn with_fullnode_run_with_range(mut self, run_with_range: RunWithRange) -> Self {
        self.fullnode_run_with_range = Some(run_with_range);
        self
    }

    pub async fn build(mut self) -> TestCluster {
        let swarm = self.start_swarm().await.unwrap();
        let working_dir = swarm.dir();

        // Find a networking validator to use as the "fullnode"
        let fullnode = swarm
            .all_nodes()
            .find(|node| {
                // It's a networking validator if it has no consensus config
                node.config().consensus_config.is_none()
            })
            .expect("No networking validator found to use as fullnode");

        let rpc_address = fullnode.config().rpc_address;
        let fullnode_handle =
            FullNodeHandle::new(fullnode.get_node_handle().unwrap(), rpc_address).await;

        let rpc_url = fullnode_handle.rpc_url.clone();

        let mut wallet_conf: SomaClientConfig =
            PersistedConfig::read(&working_dir.join(SOMA_CLIENT_CONFIG)).unwrap();
        wallet_conf.envs.push(SomaEnv {
            alias: "localnet".to_string(),
            rpc: rpc_url,
            basic_auth: None,
            chain_id: None,
        });
        wallet_conf.active_env = Some("localnet".to_string());

        wallet_conf.persisted(&working_dir.join(SOMA_CLIENT_CONFIG)).save().unwrap();

        let wallet_conf = swarm.dir().join(SOMA_CLIENT_CONFIG);
        let wallet = WalletContext::new(&wallet_conf).unwrap();

        TestCluster { wallet, fullnode_handle, swarm }
    }

    async fn start_swarm(&mut self) -> Result<Swarm, anyhow::Error> {
        // Start a mock scoring server so validators can run the audit service.
        let scoring_host = local_ip_utils::get_new_ip();
        let scoring_port = local_ip_utils::get_available_port(&scoring_host);
        let scoring_addr: SocketAddr = format!("{scoring_host}:{scoring_port}").parse().unwrap();

        let listener = tokio::net::TcpListener::bind(scoring_addr).await?;
        tokio::spawn(async move {
            scoring::tonic::transport::Server::builder()
                .add_service(scoring::tonic_gen::scoring_server::ScoringServer::new(
                    MockScoringService,
                ))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .expect("mock scoring server");
        });

        let scoring_url = format!("http://{scoring_addr}");
        info!("Mock scoring server started at {scoring_url}");

        let mut builder: SwarmBuilder =
            Swarm::builder().with_fullnode_count(1).with_scoring_url(scoring_url);

        if let Some(validators) = self.validators.take() {
            builder = builder.with_validators(validators);
        } else {
            builder = builder.committee_size(
                NonZeroUsize::new(self.num_validators.unwrap_or(NUM_VALIDATORS)).unwrap(),
            )
        };

        if let Some(genesis_config) = self.genesis_config.take() {
            builder = builder.with_genesis_config(genesis_config);
        }

        if let Some(network_config) = self.network_config.take() {
            builder = builder.with_network_config(network_config);
        }

        let mut builder = builder.with_supported_protocol_versions_config(
            self.validator_supported_protocol_versions_config.clone(),
        );

        if let Some(run_with_range) = self.fullnode_run_with_range {
            builder = builder.with_fullnode_run_with_range(run_with_range);
        }

        let mut swarm = builder.build();
        swarm.launch().await?;

        let dir = swarm.dir();

        let network_path = dir.join(SOMA_NETWORK_CONFIG);
        let wallet_path = dir.join(SOMA_CLIENT_CONFIG);
        let keystore_path = dir.join(SOMA_KEYSTORE_FILENAME);

        swarm.config().save(network_path)?;
        let mut keystore = Keystore::from(FileBasedKeystore::load_or_create(&keystore_path)?);
        for key in &swarm.config().account_keys {
            keystore.import(None, key.copy()).await?;
        }

        let active_address = keystore.addresses().first().cloned();

        // Create wallet config with stated authorities port
        SomaClientConfig {
            keystore: Keystore::from(FileBasedKeystore::load_or_create(&keystore_path)?),
            external_keys: None,
            envs: Default::default(),
            active_address,
            active_env: Default::default(),
        }
        .save(wallet_path)?;

        Ok(swarm)
    }

    fn get_or_init_genesis_config(&mut self) -> &mut GenesisConfig {
        if self.genesis_config.is_none() {
            self.genesis_config = Some(GenesisConfig::for_local_testing());
        }
        self.genesis_config.as_mut().unwrap()
    }

    pub fn with_genesis_models(mut self, models: Vec<GenesisModelConfig>) -> Self {
        self.get_or_init_genesis_config().genesis_models = models;
        self
    }

    pub fn with_validator_candidates(
        mut self,
        addresses: impl IntoIterator<Item = SomaAddress>,
    ) -> Self {
        self.get_or_init_genesis_config().accounts.extend(addresses.into_iter().map(|address| {
            AccountConfig { address: Some(address), gas_amounts: vec![DEFAULT_GAS_AMOUNT * 10] }
        }));
        self
    }
}

impl Default for TestClusterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_checksum(data: &[u8]) -> Checksum {
    let mut hasher = DefaultHash::default();
    hasher.update(data);
    Checksum::new_from_hash(hasher.finalize().into())
}
