use fastcrypto::encoding::{Encoding, Hex};
use std::{net::SocketAddr, path::PathBuf};
use types::{
    config::{
        node_config::{AuthorityKeyPairWithPath, ConsensusConfig, KeyPairWithPath, NodeConfig},
        p2p_config::{P2pConfig, SeedPeer},
    },
    crypto::{
        AuthorityKeyPair, AuthorityPublicKeyBytes, KeypairTraits, NetworkKeyPair, SomaKeyPair,
    },
    genesis::{self, Genesis},
    multiaddr::Multiaddr,
    peer_id::PeerId,
};

use super::{
    genesis_config::{ValidatorGenesisConfig, ValidatorGenesisConfigBuilder},
    network_config::NetworkConfig,
};

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

    pub fn build(self, validator: ValidatorGenesisConfig, genesis: genesis::Genesis) -> NodeConfig {
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

        let consensus_config = ConsensusConfig {
            address: consensus_address,
            db_path: consensus_db_path,
            db_pruner_period_secs: None,
            db_retention_epochs: None,
            submit_delay_step_override_millis: None,
            max_submit_position: None,
            max_pending_transactions: None,
            parameters: None,
        };

        let p2p_config = P2pConfig {
            // listen_address: Some(validator.p2p_listen_address),
            external_address: Some(validator.p2p_address),
            // Set a shorter timeout for checkpoint content download in tests, since
            // checkpoint pruning also happens much faster, and network is local.
            // state_sync: Some(StateSyncConfig {
            //     checkpoint_content_timeout_ms: Some(10_000),
            //     ..Default::default()
            // }),
            ..Default::default()
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
            network_address,
            genesis: genesis,
            consensus_config: Some(consensus_config),
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

#[derive(Clone, Debug, Default)]
pub struct FullnodeConfigBuilder {
    config_directory: Option<PathBuf>,
    db_path: Option<PathBuf>,
    network_address: Option<Multiaddr>,
    genesis: Option<Genesis>,
    network_key_pair: Option<KeyPairWithPath>,
    p2p_external_address: Option<Multiaddr>,
    // p2p_listen_address: Option<Multiaddr>,
}

impl FullnodeConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config_directory(mut self, config_directory: PathBuf) -> Self {
        self.config_directory = Some(config_directory);
        self
    }

    pub fn with_db_path(mut self, db_path: PathBuf) -> Self {
        self.db_path = Some(db_path);
        self
    }

    pub fn with_network_address(mut self, network_address: Multiaddr) -> Self {
        self.network_address = Some(network_address);
        self
    }

    pub fn with_genesis(mut self, genesis: Genesis) -> Self {
        self.genesis = Some(genesis);
        self
    }

    pub fn with_p2p_external_address(mut self, p2p_external_address: Multiaddr) -> Self {
        self.p2p_external_address = Some(p2p_external_address);
        self
    }

    // pub fn with_p2p_listen_address(mut self, p2p_listen_address: Multiaddr) -> Self {
    //     self.p2p_listen_address = Some(p2p_listen_address);
    //     self
    // }

    pub fn with_network_key_pair(mut self, network_key_pair: Option<NetworkKeyPair>) -> Self {
        if let Some(network_key_pair) = network_key_pair {
            self.network_key_pair = Some(KeyPairWithPath::new(SomaKeyPair::Ed25519(
                network_key_pair.into_inner(),
            )));
        }
        self
    }

    pub fn build<R: rand::RngCore + rand::CryptoRng>(
        self,
        rng: &mut R,
        network_config: &NetworkConfig,
    ) -> NodeConfig {
        // Take advantage of ValidatorGenesisConfigBuilder to build the keypairs and addresses,
        // even though this is a fullnode.
        let validator_config = ValidatorGenesisConfigBuilder::new().build(rng);
        let ip = validator_config
            .network_address
            .to_socket_addr()
            .unwrap()
            .ip()
            .to_string();

        let key_path = get_key_path(&validator_config.key_pair);
        let config_directory = self
            .config_directory
            .unwrap_or_else(|| tempfile::tempdir().unwrap().into_path());

        let p2p_config = {
            let seed_peers = network_config
                .validator_configs
                .iter()
                .map(|config| SeedPeer {
                    peer_id: Some(PeerId(
                        config.network_key_pair().public().into_inner().0.to_bytes(),
                    )),
                    address: config.p2p_config.external_address.clone().unwrap(),
                })
                .collect();

            P2pConfig {
                // listen_address: Some(
                //     self.p2p_listen_address
                //         .unwrap_or_else(|| validator_config.p2p_listen_address),
                // ),
                external_address: self
                    .p2p_external_address
                    .or(Some(validator_config.p2p_address.clone())),
                seed_peers,
                // Set a shorter timeout for checkpoint content download in tests, since
                // checkpoint pruning also happens much faster, and network is local.
                // state_sync: Some(StateSyncConfig {
                //     checkpoint_content_timeout_ms: Some(10_000),
                //     ..Default::default()
                // }),
                ..Default::default()
            }
        };

        NodeConfig {
            protocol_key_pair: AuthorityKeyPairWithPath::new(validator_config.key_pair),
            account_key_pair: KeyPairWithPath::new(validator_config.account_key_pair),
            worker_key_pair: KeyPairWithPath::new(SomaKeyPair::Ed25519(
                validator_config.worker_key_pair.into_inner(),
            )),
            network_key_pair: self.network_key_pair.unwrap_or(KeyPairWithPath::new(
                SomaKeyPair::Ed25519(validator_config.network_key_pair.into_inner()),
            )),
            db_path: self
                .db_path
                .unwrap_or(config_directory.join(FULL_NODE_DB_PATH).join(key_path)),
            network_address: self
                .network_address
                .unwrap_or(validator_config.network_address),
            consensus_config: None,
            genesis: self.genesis.unwrap_or(network_config.genesis.clone()),
            end_of_epoch_broadcast_channel_capacity: 128,
            p2p_config,
        }
    }
}

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
