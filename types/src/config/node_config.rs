use crate::{
    base::SomaAddress,
    config::{local_ip_utils, rpc_config::RpcConfig},
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
    net::SocketAddr,
    ops::Mul,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::Duration,
};

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

    pub end_of_epoch_broadcast_channel_capacity: usize, // 128

    #[serde(default)]
    pub p2p_config: P2pConfig,

    pub encoder_validator_address: Multiaddr,

    #[serde(default = "default_rpc_address")]
    pub rpc_address: SocketAddr,
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
                    commit_content_timeout_ms: Some(10_000),
                    ..Default::default()
                }),
                ..Default::default()
            }
        };

        NodeConfig {
            protocol_key_pair: AuthorityKeyPairWithPath::new(validator.key_pair),
            network_key_pair: KeyPairWithPath::new(SomaKeyPair::Ed25519(
                validator.network_key_pair.into_inner(),
            )),
            account_key_pair: KeyPairWithPath::new(validator.account_key_pair),
            worker_key_pair: KeyPairWithPath::new(SomaKeyPair::Ed25519(
                validator.worker_key_pair.into_inner(),
            )),
            db_path,
            consensus_db_path,
            network_address,
            genesis: genesis,
            encoder_validator_address: validator.encoder_validator_address,
            rpc_address: validator.rpc_address.to_socket_addr().unwrap(),
            rpc: Some(RpcConfig {
                ..Default::default()
            }),
            consensus_config,
            end_of_epoch_broadcast_channel_capacity: 128,
            p2p_config,
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
fn get_key_path(key_pair: &AuthorityKeyPair) -> String {
    let public_key: AuthorityPublicKeyBytes = key_pair.public().into();
    let mut key_path = Hex::encode(public_key);
    // 12 is rather arbitrary here but it's a nice balance between being short and being unique.
    key_path.truncate(12);
    key_path
}

pub const CONSENSUS_DB_NAME: &str = "consensus_db";
pub const FULL_NODE_DB_PATH: &str = "full_node_db";
pub const AUTHORITIES_DB_NAME: &str = "authorities_db";
