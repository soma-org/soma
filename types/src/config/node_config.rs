use crate::{
    base::SomaAddress,
    checkpoints::CheckpointSequenceNumber,
    committee::EpochId,
    config::{
        Config, certificate_deny_config::CertificateDenyConfig, local_ip_utils,
        object_store_config::ObjectStoreConfig, rpc_config::RpcConfig,
        transaction_deny_config::TransactionDenyConfig,
        validator_client_monitor_config::ValidatorClientMonitorConfig,
    },
    crypto::{AuthorityKeyPair, AuthorityPublicKeyBytes, NetworkKeyPair, SomaKeyPair},
    multiaddr::Multiaddr,
    parameters::Parameters,
    peer_id::PeerId,
    supported_protocol_versions::SupportedProtocolVersions,
};
use anyhow::Result;
use anyhow::anyhow;
use fastcrypto::{
    encoding::{Encoding, Hex},
    traits::{EncodeDecodeBase64, KeyPair},
};
use protocol_config::Chain;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{
    collections::BTreeMap,
    net::SocketAddr,
    num::NonZeroUsize,
    ops::Mul,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::Duration,
};
use tracing::info;

use super::{
    genesis_config::{ValidatorGenesisConfig, ValidatorGenesisConfigBuilder},
    network_config::NetworkConfig,
    p2p_config::{P2pConfig, SeedPeer},
    state_sync_config::StateSyncConfig,
};

/// Default commission rate of 2%
pub const DEFAULT_COMMISSION_RATE: u64 = 200;

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
// #[serde(rename_all = "kebab-case")]
pub struct NodeConfig {
    // #[serde(default = "default_authority_key_pair")]
    pub protocol_key_pair: AuthorityKeyPairWithPath,
    // #[serde(default = "default_key_pair")]
    pub worker_key_pair: KeyPairWithPath,
    // #[serde(default = "default_key_pair")]
    pub account_key_pair: KeyPairWithPath,
    // #[serde(default = "default_key_pair")]
    pub network_key_pair: KeyPairWithPath,

    pub db_path: PathBuf,

    pub consensus_db_path: PathBuf,
    // #[serde(default = "default_grpc_address")]
    pub network_address: Multiaddr,

    // #[serde(skip_serializing_if = "Option::is_none")]
    pub consensus_config: Option<ConsensusConfig>,

    pub rpc: Option<RpcConfig>,

    pub genesis: Genesis,

    pub authority_store_pruning_config: AuthorityStorePruningConfig,

    pub end_of_epoch_broadcast_channel_capacity: usize, // 128

    #[serde(default)]
    pub state_archive_read_config: Option<ArchiveReaderConfig>,

    #[serde(default)]
    pub p2p_config: P2pConfig,

    #[serde(default = "default_rpc_address")]
    pub rpc_address: SocketAddr,

    #[serde(default)]
    pub checkpoint_executor_config: CheckpointExecutorConfig,

    /// In a `soma-node` binary, this is set to SupportedProtocolVersions::SYSTEM_DEFAULT
    /// in node-node/src/main.rs. It is present in the config so that it can be changed by tests in
    /// order to test protocol upgrades.
    #[serde(skip)]
    pub supported_protocol_versions: Option<SupportedProtocolVersions>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_with_range: Option<RunWithRange>,

    #[serde(default)]
    pub execution_cache: ExecutionCacheConfig,

    #[serde(default)]
    pub state_debug_dump_config: StateDebugDumpConfig,

    /// Fork recovery configuration for handling validator equivocation after forks
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fork_recovery: Option<ForkRecoveryConfig>,

    #[serde(default)]
    pub transaction_deny_config: TransactionDenyConfig,

    #[serde(default)]
    pub certificate_deny_config: CertificateDenyConfig,

    #[serde(default)]
    pub expensive_safety_check_config: ExpensiveSafetyCheckConfig,

    /// Allow overriding the chain for testing purposes. For instance, it allows you to
    /// create a test network that believes it is mainnet or testnet. Attempting to
    /// override this value on production networks will result in an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain_override_for_testing: Option<Chain>,

    /// Configuration for validator client monitoring from the client perspective.
    /// When enabled, tracks client-observed performance metrics for validators.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validator_client_monitor_config: Option<ValidatorClientMonitorConfig>,

    /// Configuration for the transaction driver.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transaction_driver_config: Option<TransactionDriverConfig>,
}

impl NodeConfig {
    pub fn protocol_key_pair(&self) -> &AuthorityKeyPair {
        self.protocol_key_pair.authority_keypair()
    }

    pub fn protocol_public_key(&self) -> AuthorityPublicKeyBytes {
        self.protocol_key_pair().public().into()
    }

    pub fn network_address(&self) -> &Multiaddr {
        &self.network_address
    }

    pub fn db_path(&self) -> PathBuf {
        self.db_path.clone()
    }

    pub fn consensus_db_path(&self) -> PathBuf {
        self.consensus_db_path.clone()
    }

    pub fn worker_key_pair(&self) -> NetworkKeyPair {
        match self.worker_key_pair.keypair() {
            SomaKeyPair::Ed25519(kp) => NetworkKeyPair::new(kp.copy()),
            other => {
                panic!("Invalid keypair type: {:?}, only Ed25519 is allowed for worker key", other)
            }
        }
    }

    pub fn network_key_pair(&self) -> NetworkKeyPair {
        match self.network_key_pair.keypair() {
            SomaKeyPair::Ed25519(kp) => NetworkKeyPair::new(kp.copy()),
            other => {
                panic!("Invalid keypair type: {:?}, only Ed25519 is allowed for network key", other)
            }
        }
    }

    pub fn consensus_config(&self) -> Option<&ConsensusConfig> {
        self.consensus_config.as_ref()
    }

    pub fn genesis(&self) -> &Genesis {
        &self.genesis
    }

    pub fn soma_address(&self) -> SomaAddress {
        (&self.account_key_pair.keypair().public()).into()
    }

    pub fn rpc(&self) -> Option<&crate::config::rpc_config::RpcConfig> {
        self.rpc.as_ref()
    }
}

impl Config for NodeConfig {}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArchiveReaderConfig {
    pub remote_store_config: ObjectStoreConfig,
    pub download_concurrency: NonZeroUsize,
    pub ingestion_url: Option<String>,
    pub remote_store_options: Vec<(String, String)>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct AuthorityStorePruningConfig {
    /// number of the latest epoch dbs to retain
    #[serde(default = "default_num_latest_epoch_dbs_to_retain")]
    pub num_latest_epoch_dbs_to_retain: usize,
    /// time interval used by the pruner to determine whether there are any epoch DBs to remove
    #[serde(default = "default_epoch_db_pruning_period_secs")]
    pub epoch_db_pruning_period_secs: u64,
    /// number of epochs to keep the latest version of objects for.
    /// Note that a zero value corresponds to an aggressive pruner.
    /// This mode is experimental and needs to be used with caution.
    /// Use `u64::MAX` to disable the pruner for the objects.
    #[serde(default)]
    pub num_epochs_to_retain: u64,
    /// pruner's runtime interval used for aggressive mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pruning_run_delay_seconds: Option<u64>,
    /// maximum number of commits in the pruning batch. Can be adjusted to increase performance
    #[serde(default = "default_max_checkpoints_in_batch")]
    pub max_checkpoints_in_batch: usize,
    /// maximum number of transaction in the pruning batch
    #[serde(default = "default_max_transactions_in_batch")]
    pub max_transactions_in_batch: usize,
    /// enables periodic background compaction for old SST files whose last modified time is
    /// older than `periodic_compaction_threshold_days` days.
    /// That ensures that all sst files eventually go through the compaction process
    #[serde(
        default = "default_periodic_compaction_threshold_days",
        skip_serializing_if = "Option::is_none"
    )]
    pub periodic_compaction_threshold_days: Option<usize>,
    /// number of epochs to keep the latest version of transactions and effects for
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_epochs_to_retain_for_checkpoints: Option<u64>,
    /// disables object tombstone pruning. We don't serialize it if it is the default value, false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub killswitch_tombstone_pruning: bool,
    #[serde(default = "default_smoothing", skip_serializing_if = "is_true")]
    pub smooth: bool,
    /// Enables the compaction filter for pruning the objects table.
    /// If disabled, a range deletion approach is used instead.
    /// While it is generally safe to switch between the two modes,
    /// switching from the compaction filter approach back to range deletion
    /// may result in some old versions that will never be pruned.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub enable_compaction_filter: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_epochs_to_retain_for_indexes: Option<u64>,
}

fn default_num_latest_epoch_dbs_to_retain() -> usize {
    3
}

fn default_epoch_db_pruning_period_secs() -> u64 {
    3600
}

fn default_max_transactions_in_batch() -> usize {
    1000
}

fn default_max_checkpoints_in_batch() -> usize {
    10
}

fn default_smoothing() -> bool {
    cfg!(not(test))
}

fn is_true(value: &bool) -> bool {
    *value
}

fn default_periodic_compaction_threshold_days() -> Option<usize> {
    Some(1)
}

impl Default for AuthorityStorePruningConfig {
    fn default() -> Self {
        Self {
            num_latest_epoch_dbs_to_retain: default_num_latest_epoch_dbs_to_retain(),
            epoch_db_pruning_period_secs: default_epoch_db_pruning_period_secs(),
            num_epochs_to_retain: 0,
            pruning_run_delay_seconds: if cfg!(msim) { Some(2) } else { None },
            max_checkpoints_in_batch: default_max_checkpoints_in_batch(),
            max_transactions_in_batch: default_max_transactions_in_batch(),
            periodic_compaction_threshold_days: None,
            num_epochs_to_retain_for_checkpoints: if cfg!(msim) { Some(2) } else { None },
            killswitch_tombstone_pruning: false,
            smooth: true,
            enable_compaction_filter: cfg!(test) || cfg!(msim),
            num_epochs_to_retain_for_indexes: None,
        }
    }
}

impl AuthorityStorePruningConfig {
    pub fn set_num_epochs_to_retain(&mut self, num_epochs_to_retain: u64) {
        self.num_epochs_to_retain = num_epochs_to_retain;
    }

    pub fn set_num_epochs_to_retain_for_checkpoints(&mut self, num_epochs_to_retain: Option<u64>) {
        self.num_epochs_to_retain_for_checkpoints = num_epochs_to_retain;
    }

    pub fn num_epochs_to_retain_for_checkpoints(&self) -> Option<u64> {
        self.num_epochs_to_retain_for_checkpoints
            // if n less than 2, coerce to 2 and log
            .map(|n| {
                if n < 2 {
                    info!(
                        "num_epochs_to_retain_for_commits must be at least 2, rounding up from {}",
                        n
                    );
                    2
                } else {
                    n
                }
            })
    }

    pub fn set_killswitch_tombstone_pruning(&mut self, killswitch_tombstone_pruning: bool) {
        self.killswitch_tombstone_pruning = killswitch_tombstone_pruning;
    }
}

/// Wrapper struct for SomaKeyPair that can be deserialized from a file path. Used by network, worker, and account keypair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyPairWithPath {
    #[serde(flatten)]
    location: KeyPairLocation,
    #[serde(skip)]
    keypair: OnceLock<Arc<SomaKeyPair>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde_as]
#[serde(untagged)]
enum KeyPairLocation {
    InPlace {
        #[serde_as(as = "Arc<KeyPairBase64>")]
        value: Arc<SomaKeyPair>,
    },
    File {
        #[serde(rename = "path")]
        path: PathBuf,
    },
}
// }

impl KeyPairWithPath {
    pub fn new(kp: SomaKeyPair) -> Self {
        let cell: OnceLock<Arc<SomaKeyPair>> = OnceLock::new();
        let arc_kp = Arc::new(kp);
        // OK to unwrap panic because authority should not start without all keypairs loaded.
        cell.set(arc_kp.clone()).expect("Failed to set keypair");
        Self { location: KeyPairLocation::InPlace { value: arc_kp }, keypair: cell }
    }

    //     pub fn new_from_path(path: PathBuf) -> Self {
    //         let cell: OnceCell<Arc<SomaKeyPair>> = OnceCell::new();
    //         // OK to unwrap panic because authority should not start without all keypairs loaded.
    //         cell.set(Arc::new(read_keypair_from_file(&path).unwrap_or_else(
    //             |e| panic!("Invalid keypair file at path {:?}: {e}", &path),
    //         )))
    //         .expect("Failed to set keypair");
    //         Self {
    //             location: KeyPairLocation::File { path },
    //             keypair: cell,
    //         }
    //     }

    pub fn keypair(&self) -> &SomaKeyPair {
        self.keypair
            .get_or_init(|| match &self.location {
                KeyPairLocation::InPlace { value } => value.clone(),
                KeyPairLocation::File { path } => {
                    // OK to unwrap panic because authority should not start without all keypairs loaded.
                    Arc::new(
                        read_keypair_from_file(path).unwrap_or_else(|e| {
                            panic!("Invalid keypair file at path {:?}: {e}", path)
                        }),
                    )
                }
            })
            .as_ref()
    }
}

/// Wrapper struct for AuthorityKeyPair that can be deserialized from a file path.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct AuthorityKeyPairWithPath {
    #[serde(flatten)]
    location: AuthorityKeyPairLocation,
    #[serde(skip)]
    keypair: OnceLock<Arc<AuthorityKeyPair>>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Eq)]
#[serde(untagged)]
enum AuthorityKeyPairLocation {
    InPlace { value: Arc<AuthorityKeyPair> },
    File { path: PathBuf },
}

impl AuthorityKeyPairWithPath {
    pub fn new(kp: AuthorityKeyPair) -> Self {
        let cell: OnceLock<Arc<AuthorityKeyPair>> = OnceLock::new();
        let arc_kp = Arc::new(kp);
        // OK to unwrap panic because authority should not start without all keypairs loaded.
        cell.set(arc_kp.clone()).expect("Failed to set authority keypair");
        Self {
            location: AuthorityKeyPairLocation::InPlace { value: arc_kp },
            keypair: cell,
        }
    }

    pub fn authority_keypair(&self) -> &AuthorityKeyPair {
        self.keypair
            .get_or_init(|| match &self.location {
                AuthorityKeyPairLocation::InPlace { value } => value.clone(),
                AuthorityKeyPairLocation::File { path } => {
                    Arc::new(
                        read_authority_keypair_from_file(path).unwrap_or_else(|e| {
                            panic!("Invalid authority keypair file at path {:?}: {e}", path)
                        }),
                    )
                }
            })
            .as_ref()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct ConsensusConfig {
    // Base consensus DB path for all epochs.
    pub db_path: PathBuf,

    // The number of epochs for which to retain the consensus DBs. Setting it to 0 will make a consensus DB getting
    // dropped as soon as system is switched to a new epoch.
    pub db_retention_epochs: Option<u64>,

    // Pruner will run on every epoch change but it will also check periodically on every `db_pruner_period_secs`
    // seconds to see if there are any epoch DBs to remove.
    pub db_pruner_period_secs: Option<u64>,

    /// Maximum number of pending transactions to submit to consensus, including those
    /// in submission wait.
    /// Default to 20_000 inflight limit, assuming 20_000 txn tps * 1 sec consensus latency.
    pub max_pending_transactions: Option<usize>,

    /// When defined caps the calculated submission position to the max_submit_position. Even if the
    /// is elected to submit from a higher position than this, it will "reset" to the max_submit_position.
    pub max_submit_position: Option<usize>,

    /// The submit delay step to consensus defined in milliseconds. When provided it will
    /// override the current back off logic otherwise the default backoff logic will be applied based
    /// on consensus latency estimates.
    pub submit_delay_step_override_millis: Option<u64>,

    pub parameters: Option<Parameters>,

    pub address: Multiaddr,

    /// Port for the proxy HTTP server that serves data/model downloads.
    /// If not set, proxy server is not started.
    /// Clients fetch submission data and model weights from this address.
    pub proxy_port: Option<u16>,
}

impl ConsensusConfig {
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn address(&self) -> &Multiaddr {
        &self.address
    }

    pub fn max_pending_transactions(&self) -> usize {
        self.max_pending_transactions.unwrap_or(20_000)
    }

    pub fn submit_delay_step_override(&self) -> Option<Duration> {
        self.submit_delay_step_override_millis.map(Duration::from_millis)
    }

    pub fn db_retention_epochs(&self) -> u64 {
        self.db_retention_epochs
            // if n less than 2, coerce to 2 and log
            .map(|n| {
                if n < 2 {
                    info!("db_retention_epochs must be at least 2, rounding up from {}", n);
                    2
                } else {
                    n
                }
            })
            .unwrap_or(2)
    }

    pub fn db_pruner_period(&self) -> Duration {
        // Default to 1 hour
        self.db_pruner_period_secs.map(Duration::from_secs).unwrap_or(Duration::from_secs(3_600))
    }

    /// Get the proxy server port for serving data/model downloads.
    pub fn proxy_port(&self) -> Option<u16> {
        self.proxy_port
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct CheckpointExecutorConfig {
    /// Upper bound on the number of checkpoints that can be concurrently executed
    ///
    /// If unspecified, this will default to `200`
    #[serde(default = "default_checkpoint_execution_max_concurrency")]
    pub checkpoint_execution_max_concurrency: usize,

    /// Number of seconds to wait for effects of a batch of transactions
    /// before logging a warning. Note that we will continue to retry
    /// indefinitely
    ///
    /// If unspecified, this will default to `10`.
    #[serde(default = "default_local_execution_timeout_sec")]
    pub local_execution_timeout_sec: u64,

    /// Optional directory used for data ingestion pipeline
    /// When specified, each executed checkpoint will be saved in a local directory for post processing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_ingestion_dir: Option<PathBuf>,
}

fn default_checkpoint_execution_max_concurrency() -> usize {
    4
}

fn default_local_execution_timeout_sec() -> u64 {
    30
}

impl Default for CheckpointExecutorConfig {
    fn default() -> Self {
        Self {
            checkpoint_execution_max_concurrency: default_checkpoint_execution_max_concurrency(),
            local_execution_timeout_sec: default_local_execution_timeout_sec(),
            data_ingestion_dir: None,
        }
    }
}

// fn default_authority_key_pair() -> AuthorityKeyPairWithPath {
//     AuthorityKeyPairWithPath::new(get_key_pair_from_rng::<AuthorityKeyPair, _>(&mut OsRng).1)
// }

// fn default_key_pair() -> KeyPairWithPath {
//     KeyPairWithPath::new(
//         get_key_pair_from_rng::<AccountKeyPair, _>(&mut OsRng)
//             .1
//             .into(),
//     )
// }

pub fn default_rpc_address() -> SocketAddr {
    use std::net::{IpAddr, Ipv4Addr};
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 9000)
}

fn default_grpc_address() -> Multiaddr {
    "/ip4/0.0.0.0/tcp/8080".parse().unwrap()
}

/// Read from file as Base64 encoded `privkey` and return a AuthorityKeyPair.
pub fn read_authority_keypair_from_file<P: AsRef<std::path::Path>>(
    path: P,
) -> anyhow::Result<AuthorityKeyPair> {
    let contents = std::fs::read_to_string(path)?;
    AuthorityKeyPair::decode_base64(contents.as_str().trim()).map_err(|e| anyhow!(e))
}

/// Read from file as Base64 encoded `flag || privkey` and return a SomaKeypair.
pub fn read_keypair_from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<SomaKeyPair> {
    let contents = std::fs::read_to_string(path)?;
    SomaKeyPair::decode_base64(contents.as_str().trim()).map_err(|e| anyhow!(e))
}

/// This builder contains information that's not included in ValidatorGenesisConfig for building
/// a validator NodeConfig. It can be used to build either a genesis validator or a new validator.
#[derive(Clone, Default)]
pub struct ValidatorConfigBuilder {
    config_directory: Option<PathBuf>,
    supported_protocol_versions: Option<SupportedProtocolVersions>,
}

impl ValidatorConfigBuilder {
    pub fn new() -> Self {
        Self { ..Default::default() }
    }

    pub fn with_config_directory(mut self, config_directory: PathBuf) -> Self {
        assert!(self.config_directory.is_none());
        self.config_directory = Some(config_directory);
        self
    }

    pub fn with_supported_protocol_versions(
        mut self,
        supported_protocol_versions: SupportedProtocolVersions,
    ) -> Self {
        assert!(self.supported_protocol_versions.is_none());
        self.supported_protocol_versions = Some(supported_protocol_versions);
        self
    }

    pub fn build(
        self,
        validator: ValidatorGenesisConfig,
        genesis: crate::genesis::Genesis,
        seed_peers: Vec<SeedPeer>,
    ) -> NodeConfig {
        let key_path = get_key_path(&validator.key_pair);
        let config_directory =
            self.config_directory.unwrap_or_else(|| tempfile::tempdir().unwrap().into_path());

        let db_path = config_directory.join(AUTHORITIES_DB_NAME).join(key_path.clone());

        let network_address = validator.network_address;
        let consensus_address = validator.consensus_address;
        let consensus_db_path = config_directory.join(CONSENSUS_DB_NAME).join(key_path);

        let consensus_config = Some(ConsensusConfig {
            db_path: consensus_db_path.clone(),
            address: consensus_address,
            db_pruner_period_secs: None,
            db_retention_epochs: None,
            submit_delay_step_override_millis: None,
            max_submit_position: None,
            max_pending_transactions: None,
            parameters: None,
            proxy_port: None,
        });

        let p2p_config = {
            P2pConfig {
                // listen_address: Some(
                //     self.p2p_listen_address
                //         .unwrap_or_else(|| validator_config.p2p_listen_address),
                // ),
                external_address: Some(validator.p2p_address.clone()),
                seed_peers,
                // Set a shorter timeout for commit content download in tests, since
                // commit pruning also happens much faster, and network is local.
                state_sync: Some(StateSyncConfig {
                    checkpoint_content_timeout_ms: Some(10_000),
                    ..Default::default()
                }),
                ..Default::default()
            }
        };

        let pruning_config = AuthorityStorePruningConfig::default();

        let checkpoint_executor_config = CheckpointExecutorConfig {
            // TODO: data_ingestion_dir: self.data_ingestion_dir,
            ..Default::default()
        };

        NodeConfig {
            protocol_key_pair: AuthorityKeyPairWithPath::new(validator.key_pair),
            network_key_pair: KeyPairWithPath::new(SomaKeyPair::Ed25519(
                validator.network_key_pair.clone().into_inner(),
            )),
            account_key_pair: KeyPairWithPath::new(validator.account_key_pair),
            worker_key_pair: KeyPairWithPath::new(SomaKeyPair::Ed25519(
                validator.worker_key_pair.clone().into_inner(),
            )),
            db_path,
            consensus_db_path,
            network_address,
            genesis: Genesis::new(genesis),
            rpc_address: validator.rpc_address.to_socket_addr().unwrap(),
            rpc: Some(RpcConfig { ..Default::default() }),
            checkpoint_executor_config,
            state_archive_read_config: None,
            consensus_config,
            authority_store_pruning_config: pruning_config,
            end_of_epoch_broadcast_channel_capacity: 128,
            p2p_config,
            state_debug_dump_config: Default::default(),
            validator_client_monitor_config: None,
            chain_override_for_testing: None,
            run_with_range: None,
            supported_protocol_versions: self.supported_protocol_versions,
            // By default, expensive checks will be enabled in debug build, but not in release build.
            expensive_safety_check_config: ExpensiveSafetyCheckConfig::default(),
            execution_cache: ExecutionCacheConfig::default(),
            fork_recovery: None,
            transaction_driver_config: Some(TransactionDriverConfig::default()),
            transaction_deny_config: Default::default(),
            certificate_deny_config: Default::default(),
        }
    }

    // pub fn build_new_validator<R: rand::RngCore + rand::CryptoRng>(
    //     self,
    //     rng: &mut R,
    //     network_config: &NetworkConfig,
    // ) -> NodeConfig {
    //     let validator_config = ValidatorGenesisConfigBuilder::new().build(rng);
    //     self.build(validator_config, network_config.genesis.clone())
    // }
}

/// Builder for fullnode NodeConfig. Creates a node that is NOT in the validator committee
/// and does NOT participate in consensus. It connects to the network via seed peers and
/// syncs state via the state sync protocol.
#[derive(Clone, Default)]
pub struct FullnodeConfigBuilder {
    config_directory: Option<PathBuf>,
    rpc_port: Option<u16>,
    rpc_addr: Option<SocketAddr>,
    rpc_config: Option<RpcConfig>,
    supported_protocol_versions: Option<SupportedProtocolVersions>,
    run_with_range: Option<RunWithRange>,
}

impl FullnodeConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config_directory(mut self, config_directory: PathBuf) -> Self {
        self.config_directory = Some(config_directory);
        self
    }

    pub fn with_rpc_port(mut self, rpc_port: u16) -> Self {
        assert!(self.rpc_addr.is_none(), "Cannot set both rpc_port and rpc_addr");
        self.rpc_port = Some(rpc_port);
        self
    }

    pub fn with_rpc_addr(mut self, rpc_addr: SocketAddr) -> Self {
        assert!(self.rpc_port.is_none(), "Cannot set both rpc_port and rpc_addr");
        self.rpc_addr = Some(rpc_addr);
        self
    }

    pub fn with_rpc_config(mut self, rpc_config: RpcConfig) -> Self {
        self.rpc_config = Some(rpc_config);
        self
    }

    pub fn with_supported_protocol_versions(
        mut self,
        supported_protocol_versions: SupportedProtocolVersions,
    ) -> Self {
        self.supported_protocol_versions = Some(supported_protocol_versions);
        self
    }

    pub fn with_run_with_range(mut self, run_with_range: RunWithRange) -> Self {
        self.run_with_range = Some(run_with_range);
        self
    }

    pub fn build(self, genesis: crate::genesis::Genesis, seed_peers: Vec<SeedPeer>) -> NodeConfig {
        use crate::crypto::get_key_pair_from_rng;
        use fastcrypto::traits::KeyPair;
        use rand::rngs::OsRng;

        let mut rng = OsRng;

        // Generate independent keypairs — this node is NOT in the validator committee
        let protocol_key_pair: AuthorityKeyPair = get_key_pair_from_rng(&mut rng).1;
        let network_key_pair: NetworkKeyPair =
            NetworkKeyPair::new(get_key_pair_from_rng(&mut rng).1);
        let worker_key_pair: NetworkKeyPair =
            NetworkKeyPair::new(get_key_pair_from_rng(&mut rng).1);
        let account_key_pair: SomaKeyPair = SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut rng).1);

        let key_path = get_key_path(&protocol_key_pair);
        let config_directory =
            self.config_directory.unwrap_or_else(|| tempfile::tempdir().unwrap().into_path());

        let db_path = config_directory.join(FULL_NODE_DB_PATH).join(&key_path);
        let consensus_db_path = config_directory.join(CONSENSUS_DB_NAME).join(&key_path);

        let ip = local_ip_utils::get_new_ip();
        let network_address = local_ip_utils::new_tcp_address_for_testing(&ip);
        let p2p_address = local_ip_utils::new_tcp_address_for_testing(&ip);

        let rpc_address = if let Some(addr) = self.rpc_addr {
            addr
        } else if let Some(port) = self.rpc_port {
            let ip_addr: std::net::IpAddr = ip.parse().unwrap();
            SocketAddr::new(ip_addr, port)
        } else {
            let ip_addr: std::net::IpAddr = ip.parse().unwrap();
            let port = local_ip_utils::get_available_port(&ip);
            SocketAddr::new(ip_addr, port)
        };

        let p2p_config = P2pConfig {
            external_address: Some(p2p_address),
            seed_peers,
            state_sync: Some(StateSyncConfig {
                checkpoint_content_timeout_ms: Some(10_000),
                ..Default::default()
            }),
            ..Default::default()
        };

        NodeConfig {
            protocol_key_pair: AuthorityKeyPairWithPath::new(protocol_key_pair),
            network_key_pair: KeyPairWithPath::new(SomaKeyPair::Ed25519(
                network_key_pair.into_inner(),
            )),
            account_key_pair: KeyPairWithPath::new(account_key_pair),
            worker_key_pair: KeyPairWithPath::new(SomaKeyPair::Ed25519(
                worker_key_pair.into_inner(),
            )),
            db_path,
            consensus_db_path,
            network_address,
            genesis: Genesis::new(genesis),
            rpc_address,
            rpc: self.rpc_config.or(Some(RpcConfig::default())),
            checkpoint_executor_config: CheckpointExecutorConfig::default(),
            state_archive_read_config: None,
            consensus_config: None, // Fullnodes do not participate in consensus
            authority_store_pruning_config: AuthorityStorePruningConfig::default(),
            end_of_epoch_broadcast_channel_capacity: 128,
            p2p_config,
            state_debug_dump_config: Default::default(),
            validator_client_monitor_config: None,
            chain_override_for_testing: None,
            run_with_range: self.run_with_range,
            supported_protocol_versions: self.supported_protocol_versions,
            expensive_safety_check_config: ExpensiveSafetyCheckConfig::default(),
            execution_cache: ExecutionCacheConfig::default(),
            fork_recovery: None,
            transaction_driver_config: Some(TransactionDriverConfig::default()),
            transaction_deny_config: Default::default(),
            certificate_deny_config: Default::default(),
        }
    }
}

/// Given a validator keypair, return a path that can be used to identify the validator.
pub fn get_key_path(key_pair: &AuthorityKeyPair) -> String {
    let public_key: AuthorityPublicKeyBytes = key_pair.public().into();
    let mut key_path = Hex::encode(public_key);
    // 12 is rather arbitrary here but it's a nice balance between being short and being unique.
    key_path.truncate(12);
    key_path
}

pub const CONSENSUS_DB_NAME: &str = "consensus_db";
pub const FULL_NODE_DB_PATH: &str = "full_node_db";
pub const AUTHORITIES_DB_NAME: &str = "authorities_db";

// RunWithRange is used to specify the ending epoch/checkpoint to process.
// this is intended for use with disaster recovery debugging and verification workflows, never in normal operations
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum RunWithRange {
    Epoch(EpochId),
    Checkpoint(CheckpointSequenceNumber),
}

impl RunWithRange {
    // is epoch_id > RunWithRange::Epoch
    pub fn is_epoch_gt(&self, epoch_id: EpochId) -> bool {
        matches!(self, RunWithRange::Epoch(e) if epoch_id > *e)
    }

    pub fn matches_checkpoint(&self, seq_num: CheckpointSequenceNumber) -> bool {
        matches!(self, RunWithRange::Checkpoint(seq) if *seq == seq_num)
    }

    pub fn into_checkpoint_bound(self) -> Option<CheckpointSequenceNumber> {
        match self {
            RunWithRange::Epoch(_) => None,
            RunWithRange::Checkpoint(seq) => Some(seq),
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionCacheConfig {
    WritebackCache {
        /// Maximum number of entries in each cache. (There are several different caches).
        /// If None, the default of 10000 is used.
        max_cache_size: Option<u64>,

        package_cache_size: Option<u64>, // defaults to 1000

        object_cache_size: Option<u64>, // defaults to max_cache_size
        marker_cache_size: Option<u64>, // defaults to object_cache_size
        object_by_id_cache_size: Option<u64>, // defaults to object_cache_size

        transaction_cache_size: Option<u64>, // defaults to max_cache_size
        executed_effect_cache_size: Option<u64>, // defaults to transaction_cache_size
        effect_cache_size: Option<u64>,      // defaults to executed_effect_cache_size

        transaction_objects_cache_size: Option<u64>, // defaults to 1000

        /// Number of uncommitted transactions at which to pause consensus handler.
        backpressure_threshold: Option<u64>,

        /// Number of uncommitted transactions at which to refuse new transaction
        /// submissions. Defaults to backpressure_threshold if unset.
        backpressure_threshold_for_rpc: Option<u64>,

        fastpath_transaction_outputs_cache_size: Option<u64>,
    },
}

impl Default for ExecutionCacheConfig {
    fn default() -> Self {
        ExecutionCacheConfig::WritebackCache {
            max_cache_size: None,
            backpressure_threshold: None,
            backpressure_threshold_for_rpc: None,
            package_cache_size: None,
            object_cache_size: None,
            marker_cache_size: None,
            object_by_id_cache_size: None,
            transaction_cache_size: None,
            executed_effect_cache_size: None,
            effect_cache_size: None,
            transaction_objects_cache_size: None,
            fastpath_transaction_outputs_cache_size: None,
        }
    }
}

impl ExecutionCacheConfig {
    pub fn max_cache_size(&self) -> u64 {
        std::env::var("SOMA_MAX_CACHE_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or_else(
            || match self {
                ExecutionCacheConfig::WritebackCache { max_cache_size, .. } => {
                    max_cache_size.unwrap_or(100000)
                }
            },
        )
    }

    pub fn package_cache_size(&self) -> u64 {
        std::env::var("SOMA_PACKAGE_CACHE_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or_else(
            || match self {
                ExecutionCacheConfig::WritebackCache { package_cache_size, .. } => {
                    package_cache_size.unwrap_or(1000)
                }
            },
        )
    }

    pub fn object_cache_size(&self) -> u64 {
        std::env::var("SOMA_OBJECT_CACHE_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or_else(
            || match self {
                ExecutionCacheConfig::WritebackCache { object_cache_size, .. } => {
                    object_cache_size.unwrap_or(self.max_cache_size())
                }
            },
        )
    }

    pub fn marker_cache_size(&self) -> u64 {
        std::env::var("SOMA_MARKER_CACHE_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or_else(
            || match self {
                ExecutionCacheConfig::WritebackCache { marker_cache_size, .. } => {
                    marker_cache_size.unwrap_or(self.object_cache_size())
                }
            },
        )
    }

    pub fn object_by_id_cache_size(&self) -> u64 {
        std::env::var("SOMA_OBJECT_BY_ID_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache { object_by_id_cache_size, .. } => {
                    object_by_id_cache_size.unwrap_or(self.object_cache_size())
                }
            })
    }

    pub fn transaction_cache_size(&self) -> u64 {
        std::env::var("SOMA_TRANSACTION_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache { transaction_cache_size, .. } => {
                    transaction_cache_size.unwrap_or(self.max_cache_size())
                }
            })
    }

    pub fn executed_effect_cache_size(&self) -> u64 {
        std::env::var("SOMA_EXECUTED_EFFECT_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache { executed_effect_cache_size, .. } => {
                    executed_effect_cache_size.unwrap_or(self.transaction_cache_size())
                }
            })
    }

    pub fn effect_cache_size(&self) -> u64 {
        std::env::var("SOMA_EFFECT_CACHE_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or_else(
            || match self {
                ExecutionCacheConfig::WritebackCache { effect_cache_size, .. } => {
                    effect_cache_size.unwrap_or(self.executed_effect_cache_size())
                }
            },
        )
    }

    pub fn transaction_objects_cache_size(&self) -> u64 {
        std::env::var("SOMA_TRANSACTION_OBJECTS_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache { transaction_objects_cache_size, .. } => {
                    transaction_objects_cache_size.unwrap_or(1000)
                }
            })
    }

    pub fn backpressure_threshold(&self) -> u64 {
        std::env::var("SOMA_BACKPRESSURE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache { backpressure_threshold, .. } => {
                    backpressure_threshold.unwrap_or(100_000)
                }
            })
    }

    pub fn backpressure_threshold_for_rpc(&self) -> u64 {
        std::env::var("SOMA_BACKPRESSURE_THRESHOLD_FOR_RPC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache { backpressure_threshold_for_rpc, .. } => {
                    backpressure_threshold_for_rpc.unwrap_or(self.backpressure_threshold())
                }
            })
    }

    pub fn fastpath_transaction_outputs_cache_size(&self) -> u64 {
        std::env::var("SOMA_FASTPATH_TRANSACTION_OUTPUTS_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    fastpath_transaction_outputs_cache_size,
                    ..
                } => fastpath_transaction_outputs_cache_size.unwrap_or(10_000),
            })
    }
}

/// Configurations which determine how we dump state debug info.
/// Debug info is dumped when a node forks.
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
#[serde(rename_all = "kebab-case")]
pub struct StateDebugDumpConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dump_file_directory: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct ExpensiveSafetyCheckConfig {
    /// If enabled, at epoch boundary, we will check that the accumulated
    /// live object state matches the end of epoch root state digest.
    #[serde(default)]
    enable_state_consistency_check: bool,

    /// Disable state consistency check even when we are running in debug mode.
    #[serde(default)]
    force_disable_state_consistency_check: bool,
}

impl ExpensiveSafetyCheckConfig {
    pub fn new_enable_all() -> Self {
        Self { enable_state_consistency_check: true, force_disable_state_consistency_check: false }
    }

    pub fn new_disable_all() -> Self {
        Self { enable_state_consistency_check: false, force_disable_state_consistency_check: true }
    }

    pub fn force_disable_state_consistency_check(&mut self) {
        self.force_disable_state_consistency_check = true;
    }

    pub fn enable_state_consistency_check(&self) -> bool {
        (self.enable_state_consistency_check || cfg!(debug_assertions))
            && !self.force_disable_state_consistency_check
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ForkCrashBehavior {
    #[serde(rename = "await-fork-recovery")]
    #[default]
    AwaitForkRecovery,
    /// Return an error instead of blocking forever. This is primarily for testing.
    #[serde(rename = "return-error")]
    ReturnError,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct ForkRecoveryConfig {
    /// Map of transaction digest to effects digest overrides
    /// Used to repoint transactions to correct effects after a fork
    #[serde(default)]
    pub transaction_overrides: BTreeMap<String, String>,

    /// Map of checkpoint sequence number to checkpoint digest overrides
    /// On node start, if we have a locally computed checkpoint with a
    /// digest mismatch with this table, we will clear any associated local state.
    #[serde(default)]
    pub checkpoint_overrides: BTreeMap<u64, String>,

    /// Behavior when a fork is detected after recovery attempts
    #[serde(default)]
    pub fork_crash_behavior: ForkCrashBehavior,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct TransactionDriverConfig {
    /// The list of validators that are allowed to submit MFP transactions to (via the transaction driver).
    /// Each entry is a validator display name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_submission_validators: Vec<String>,

    /// The list of validators that are blocked from submitting block transactions to (via the transaction driver).
    /// Each entry is a validator display name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocked_submission_validators: Vec<String>,

    /// Enable early transaction validation before submission to consensus.
    /// This checks for non-retriable errors (like old object versions) and rejects
    /// transactions early to provide fast feedback to clients.
    /// Note: Currently used in TransactionOrchestrator, but may be moved to TransactionDriver in future.
    #[serde(default = "bool_true")]
    pub enable_early_validation: bool,
}

impl Default for TransactionDriverConfig {
    fn default() -> Self {
        Self {
            allowed_submission_validators: vec![],
            blocked_submission_validators: vec![],
            enable_early_validation: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Genesis {
    #[serde(flatten)]
    location: GenesisLocation,

    #[serde(skip)]
    genesis: once_cell::sync::OnceCell<crate::genesis::Genesis>,
}

impl Genesis {
    pub fn new(genesis: crate::genesis::Genesis) -> Self {
        Self { location: GenesisLocation::InPlace { genesis }, genesis: Default::default() }
    }

    pub fn new_from_file<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            location: GenesisLocation::File { genesis_file_location: path.into() },
            genesis: Default::default(),
        }
    }

    pub fn genesis(&self) -> Result<&crate::genesis::Genesis> {
        match &self.location {
            GenesisLocation::InPlace { genesis } => Ok(genesis),
            GenesisLocation::File { genesis_file_location } => self
                .genesis
                .get_or_try_init(|| crate::genesis::Genesis::load(genesis_file_location)),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
enum GenesisLocation {
    InPlace {
        genesis: crate::genesis::Genesis,
    },
    File {
        #[serde(rename = "genesis-file-location")]
        genesis_file_location: PathBuf,
    },
}

pub fn bool_true() -> bool {
    true
}

pub fn default_json_rpc_address() -> SocketAddr {
    use std::net::{IpAddr, Ipv4Addr};
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 9000)
}
