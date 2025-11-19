use crate::{
    base::SomaAddress,
    checkpoints::CheckpointSequenceNumber,
    committee::EpochId,
    config::{local_ip_utils, object_store_config::ObjectStoreConfig, rpc_config::RpcConfig},
    crypto::{AuthorityKeyPair, AuthorityPublicKeyBytes, NetworkKeyPair, SomaKeyPair},
    genesis::Genesis,
    multiaddr::Multiaddr,
    parameters::Parameters,
    peer_id::PeerId,
};
use anyhow::anyhow;
use fastcrypto::{
    encoding::{Encoding, Hex},
    traits::{EncodeDecodeBase64, KeyPair},
};
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

    pub encoder_validator_address: Multiaddr,

    /// The network address for the unencrypted object storage
    pub internal_object_address: Multiaddr,
    /// The network address for object storage
    pub external_object_address: Multiaddr,

    #[serde(default = "default_rpc_address")]
    pub rpc_address: SocketAddr,

    #[serde(default)]
    pub checkpoint_executor_config: CheckpointExecutorConfig,

    #[serde(default)]
    pub execution_cache: ExecutionCacheConfig,

    #[serde(default)]
    pub state_debug_dump_config: StateDebugDumpConfig,

    /// Fork recovery configuration for handling validator equivocation after forks
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fork_recovery: Option<ForkRecoveryConfig>,

    // #[serde(default)]
    // pub transaction_deny_config: TransactionDenyConfig,

    // #[serde(default)]
    // pub certificate_deny_config: CertificateDenyConfig,
    #[serde(default)]
    pub expensive_safety_check_config: ExpensiveSafetyCheckConfig,
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
            other => panic!(
                "Invalid keypair type: {:?}, only Ed25519 is allowed for worker key",
                other
            ),
        }
    }

    pub fn network_key_pair(&self) -> NetworkKeyPair {
        match self.network_key_pair.keypair() {
            SomaKeyPair::Ed25519(kp) => NetworkKeyPair::new(kp.copy()),
            other => panic!(
                "Invalid keypair type: {:?}, only Ed25519 is allowed for network key",
                other
            ),
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

    pub fn encoder_validator_address(&self) -> &Multiaddr {
        &self.encoder_validator_address
    }

    pub fn rpc(&self) -> Option<&crate::config::rpc_config::RpcConfig> {
        self.rpc.as_ref()
    }
}

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
        Self {
            location: KeyPairLocation::InPlace { value: arc_kp },
            keypair: cell,
        }
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
    // #[serde(flatten)]
    // location: AuthorityKeyPairLocation,
    #[serde(skip)]
    keypair: OnceLock<Arc<AuthorityKeyPair>>,
}

// #[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Eq)]
// #[serde_as]
// #[serde(untagged)]
// enum AuthorityKeyPairLocation {
//     InPlace { value: Arc<AuthorityKeyPair> },
//     File { path: PathBuf },
// }

impl AuthorityKeyPairWithPath {
    pub fn new(kp: AuthorityKeyPair) -> Self {
        let cell: OnceLock<Arc<AuthorityKeyPair>> = OnceLock::new();
        let arc_kp = Arc::new(kp);
        // OK to unwrap panic because authority should not start without all keypairs loaded.
        cell.set(arc_kp.clone())
            .expect("Failed to set authority keypair");
        Self { keypair: cell }
    }

    pub fn authority_keypair(&self) -> &AuthorityKeyPair {
        self.keypair.get().unwrap().as_ref()
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
        self.submit_delay_step_override_millis
            .map(Duration::from_millis)
    }

    pub fn db_retention_epochs(&self) -> u64 {
        self.db_retention_epochs
            // if n less than 2, coerce to 2 and log
            .map(|n| {
                if n < 2 {
                    info!(
                        "db_retention_epochs must be at least 2, rounding up from {}",
                        n
                    );
                    2
                } else {
                    n
                }
            })
            .unwrap_or(2)
    }

    pub fn db_pruner_period(&self) -> Duration {
        // Default to 1 hour
        self.db_pruner_period_secs
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(3_600))
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
}

impl ValidatorConfigBuilder {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn with_config_directory(mut self, config_directory: PathBuf) -> Self {
        assert!(self.config_directory.is_none());
        self.config_directory = Some(config_directory);
        self
    }

    pub fn build(
        self,
        validator: ValidatorGenesisConfig,
        genesis: Genesis,
        seed_peers: Vec<SeedPeer>,
    ) -> NodeConfig {
        let key_path = get_key_path(&validator.key_pair);
        let config_directory = self
            .config_directory
            .unwrap_or_else(|| tempfile::tempdir().unwrap().into_path());

        let db_path = config_directory
            .join(AUTHORITIES_DB_NAME)
            .join(key_path.clone());

        let network_address = validator.network_address;
        let consensus_address = validator.consensus_address;
        let consensus_db_path = config_directory.join(CONSENSUS_DB_NAME).join(key_path);

        let consensus_config = if validator.is_networking_only {
            None
        } else {
            Some(ConsensusConfig {
                db_path: consensus_db_path.clone(),
                address: consensus_address,
                db_pruner_period_secs: None,
                db_retention_epochs: None,
                submit_delay_step_override_millis: None,
                max_submit_position: None,
                max_pending_transactions: None,
                parameters: None,
            })
        };

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
            genesis: genesis,
            encoder_validator_address: validator.encoder_validator_address,
            rpc_address: validator.rpc_address.to_socket_addr().unwrap(),
            internal_object_address: validator.internal_object_address,
            external_object_address: validator.external_object_address,
            rpc: Some(RpcConfig {
                ..Default::default()
            }),
            checkpoint_executor_config,
            state_archive_read_config: None,
            consensus_config,
            authority_store_pruning_config: pruning_config,
            end_of_epoch_broadcast_channel_capacity: 128,
            p2p_config,
            state_debug_dump_config: Default::default(),
            // By default, expensive checks will be enabled in debug build, but not in release build.
            expensive_safety_check_config: ExpensiveSafetyCheckConfig::default(),
            execution_cache: ExecutionCacheConfig::default(),
            fork_recovery: None,
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

// #[derive(Clone, Debug, Default)]
// pub struct FullnodeConfigBuilder {
//     config_directory: Option<PathBuf>,
//     db_path: Option<PathBuf>,
//     network_address: Option<Multiaddr>,
//     genesis: Option<Genesis>,
//     network_key_pair: Option<KeyPairWithPath>,
//     p2p_external_address: Option<Multiaddr>,
//     encoder_validator_address: Option<Multiaddr>,
//     // p2p_listen_address: Option<Multiaddr>,
// }

// impl FullnodeConfigBuilder {
//     pub fn new() -> Self {
//         Self::default()
//     }

//     pub fn with_config_directory(mut self, config_directory: PathBuf) -> Self {
//         self.config_directory = Some(config_directory);
//         self
//     }

//     pub fn with_db_path(mut self, db_path: PathBuf) -> Self {
//         self.db_path = Some(db_path);
//         self
//     }

//     pub fn with_network_address(mut self, network_address: Multiaddr) -> Self {
//         self.network_address = Some(network_address);
//         self
//     }

//     pub fn with_genesis(mut self, genesis: Genesis) -> Self {
//         self.genesis = Some(genesis);
//         self
//     }

//     pub fn with_p2p_external_address(mut self, p2p_external_address: Multiaddr) -> Self {
//         self.p2p_external_address = Some(p2p_external_address);
//         self
//     }

//     pub fn with_encoder_validator_address(mut self, encoder_validator_address: Multiaddr) -> Self {
//         self.encoder_validator_address = Some(encoder_validator_address);
//         self
//     }

//     // pub fn with_p2p_listen_address(mut self, p2p_listen_address: Multiaddr) -> Self {
//     //     self.p2p_listen_address = Some(p2p_listen_address);
//     //     self
//     // }

//     pub fn with_network_key_pair(mut self, network_key_pair: Option<NetworkKeyPair>) -> Self {
//         if let Some(network_key_pair) = network_key_pair {
//             self.network_key_pair = Some(KeyPairWithPath::new(SomaKeyPair::Ed25519(
//                 network_key_pair.into_inner(),
//             )));
//         }
//         self
//     }

//     pub fn build<R: rand::RngCore + rand::CryptoRng>(
//         self,
//         rng: &mut R,
//         network_config: &NetworkConfig,
//     ) -> NodeConfig {
//         // Take advantage of ValidatorGenesisConfigBuilder to build the keypairs and addresses,
//         // even though this is a fullnode.
//         let validator_config = ValidatorGenesisConfigBuilder::new().build(rng);
//         let ip = validator_config
//             .network_address
//             .to_socket_addr()
//             .unwrap()
//             .ip()
//             .to_string();

//         let key_path = get_key_path(&validator_config.key_pair);
//         let config_directory = self
//             .config_directory
//             .unwrap_or_else(|| tempfile::tempdir().unwrap().into_path());

//         let p2p_config = {
//             let seed_peers = network_config
//                 .validator_configs
//                 .iter()
//                 .map(|config| SeedPeer {
//                     peer_id: Some(PeerId(
//                         config.network_key_pair().public().into_inner().0.to_bytes(),
//                     )),
//                     address: config.p2p_config.external_address.clone().unwrap(),
//                 })
//                 .collect();

//             P2pConfig {
//                 // listen_address: Some(
//                 //     self.p2p_listen_address
//                 //         .unwrap_or_else(|| validator_config.p2p_listen_address),
//                 // ),
//                 external_address: self
//                     .p2p_external_address
//                     .or(Some(validator_config.p2p_address.clone())),
//                 seed_peers,
//                 // Set a shorter timeout for commit content download in tests, since
//                 // commit pruning also happens much faster, and network is local.
//                 state_sync: Some(StateSyncConfig {
//                     commit_content_timeout_ms: Some(10_000),
//                     ..Default::default()
//                 }),
//                 ..Default::default()
//             }
//         };

//         NodeConfig {
//             protocol_key_pair: AuthorityKeyPairWithPath::new(validator_config.key_pair),
//             account_key_pair: KeyPairWithPath::new(validator_config.account_key_pair),
//             worker_key_pair: KeyPairWithPath::new(SomaKeyPair::Ed25519(
//                 validator_config.worker_key_pair.into_inner(),
//             )),
//             network_key_pair: self.network_key_pair.unwrap_or(KeyPairWithPath::new(
//                 SomaKeyPair::Ed25519(validator_config.network_key_pair.into_inner()),
//             )),
//             db_path: self
//                 .db_path
//                 .unwrap_or(config_directory.join(FULL_NODE_DB_PATH).join(key_path)),
//             network_address: self
//                 .network_address
//                 .unwrap_or(validator_config.network_address),
//             consensus_config: None,
//             encoder_validator_address: self
//                 .encoder_validator_address
//                 .unwrap_or(validator_config.encoder_validator_address),
//             genesis: self.genesis.unwrap_or(network_config.genesis.clone()),
//             end_of_epoch_broadcast_channel_capacity: 128,
//             p2p_config,
//         }
//     }
// }

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
pub const ENCODERS_DB_NAME: &str = "encoders_db";

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
        std::env::var("SOMA_MAX_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache { max_cache_size, .. } => {
                    max_cache_size.unwrap_or(100000)
                }
            })
    }

    pub fn package_cache_size(&self) -> u64 {
        std::env::var("SOMA_PACKAGE_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    package_cache_size, ..
                } => package_cache_size.unwrap_or(1000),
            })
    }

    pub fn object_cache_size(&self) -> u64 {
        std::env::var("SOMA_OBJECT_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    object_cache_size, ..
                } => object_cache_size.unwrap_or(self.max_cache_size()),
            })
    }

    pub fn marker_cache_size(&self) -> u64 {
        std::env::var("SOMA_MARKER_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    marker_cache_size, ..
                } => marker_cache_size.unwrap_or(self.object_cache_size()),
            })
    }

    pub fn object_by_id_cache_size(&self) -> u64 {
        std::env::var("SOMA_OBJECT_BY_ID_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    object_by_id_cache_size,
                    ..
                } => object_by_id_cache_size.unwrap_or(self.object_cache_size()),
            })
    }

    pub fn transaction_cache_size(&self) -> u64 {
        std::env::var("SOMA_TRANSACTION_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    transaction_cache_size,
                    ..
                } => transaction_cache_size.unwrap_or(self.max_cache_size()),
            })
    }

    pub fn executed_effect_cache_size(&self) -> u64 {
        std::env::var("SOMA_EXECUTED_EFFECT_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    executed_effect_cache_size,
                    ..
                } => executed_effect_cache_size.unwrap_or(self.transaction_cache_size()),
            })
    }

    pub fn effect_cache_size(&self) -> u64 {
        std::env::var("SOMA_EFFECT_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    effect_cache_size, ..
                } => effect_cache_size.unwrap_or(self.executed_effect_cache_size()),
            })
    }

    pub fn transaction_objects_cache_size(&self) -> u64 {
        std::env::var("SOMA_TRANSACTION_OBJECTS_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    transaction_objects_cache_size,
                    ..
                } => transaction_objects_cache_size.unwrap_or(1000),
            })
    }

    pub fn backpressure_threshold(&self) -> u64 {
        std::env::var("SOMA_BACKPRESSURE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    backpressure_threshold,
                    ..
                } => backpressure_threshold.unwrap_or(100_000),
            })
    }

    pub fn backpressure_threshold_for_rpc(&self) -> u64 {
        std::env::var("SOMA_BACKPRESSURE_THRESHOLD_FOR_RPC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| match self {
                ExecutionCacheConfig::WritebackCache {
                    backpressure_threshold_for_rpc,
                    ..
                } => backpressure_threshold_for_rpc.unwrap_or(self.backpressure_threshold()),
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
        Self {
            enable_state_consistency_check: true,
            force_disable_state_consistency_check: false,
        }
    }

    pub fn new_disable_all() -> Self {
        Self {
            enable_state_consistency_check: false,
            force_disable_state_consistency_check: true,
        }
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
