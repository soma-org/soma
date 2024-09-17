use std::num::NonZeroUsize;

use fastcrypto::traits::KeyPair;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use types::{
    committee::{Committee, CommitteeWithNetworkMetadata},
    crypto::{get_key_pair_from_rng, SomaKeyPair},
    genesis::{self, Genesis},
    multiaddr::Multiaddr,
    node_config::NodeConfig,
    system_state::{SystemParameters, SystemState, Validator},
    transaction::VerifiedTransaction,
};

use super::{
    genesis_config::{GenesisConfig, ValidatorGenesisConfigBuilder},
    node_config_builder::ValidatorConfigBuilder,
    CommitteeConfig,
};

#[derive(Debug, Deserialize, Serialize)]
pub struct NetworkConfig {
    pub validator_configs: Vec<NodeConfig>,
    pub account_keys: Vec<SomaKeyPair>,
    pub genesis: genesis::Genesis,
}

impl NetworkConfig {
    pub fn validator_configs(&self) -> &[NodeConfig] {
        &self.validator_configs
    }

    pub fn net_addresses(&self) -> Vec<Multiaddr> {
        self.genesis
            .committee_with_network()
            .validators()
            .values()
            .map(|(_, n)| n.network_address.clone())
            .collect()
    }

    pub fn committee_with_network(&self) -> CommitteeWithNetworkMetadata {
        self.genesis.committee_with_network()
    }

    pub fn into_validator_configs(self) -> Vec<NodeConfig> {
        self.validator_configs
    }
}

pub struct ConfigBuilder<R = OsRng> {
    rng: Option<R>,
    committee: CommitteeConfig,
    genesis_config: Option<GenesisConfig>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            rng: Some(OsRng),
            committee: CommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
            genesis_config: None,
        }
    }
}

impl<R> ConfigBuilder<R> {
    pub fn committee(mut self, committee: CommitteeConfig) -> Self {
        self.committee = committee;
        self
    }

    pub fn committee_size(mut self, committee_size: NonZeroUsize) -> Self {
        self.committee = CommitteeConfig::Size(committee_size);
        self
    }

    pub fn with_genesis_config(mut self, genesis_config: GenesisConfig) -> Self {
        assert!(self.genesis_config.is_none(), "Genesis config already set");
        self.genesis_config = Some(genesis_config);
        self
    }

    pub fn rng<N: rand::RngCore + rand::CryptoRng>(self, rng: N) -> ConfigBuilder<N> {
        ConfigBuilder {
            rng: Some(rng),
            committee: self.committee,
            genesis_config: self.genesis_config,
        }
    }

    fn get_or_init_genesis_config(&mut self) -> &mut GenesisConfig {
        if self.genesis_config.is_none() {
            self.genesis_config = Some(GenesisConfig::for_local_testing());
        }
        self.genesis_config.as_mut().unwrap()
    }
}

impl<R: rand::RngCore + rand::CryptoRng> ConfigBuilder<R> {
    //TODO right now we always randomize ports, we may want to have a default port configuration
    pub fn build(self) -> NetworkConfig {
        let committee = self.committee;

        let mut rng = self.rng.unwrap();
        let validators = match committee {
            CommitteeConfig::Size(size) => {
                // We always get fixed protocol keys from this function (which is isolated from
                // external test randomness because it uses a fixed seed). Necessary because some
                // tests call `make_tx_certs_and_signed_effects`, which locally forges a cert using
                // this same committee.
                let (_, keys) = Committee::new_simple_test_committee_of_size(size.into());

                keys.into_iter()
                    .map(|authority_key| {
                        let mut builder = ValidatorGenesisConfigBuilder::new()
                            .with_protocol_key_pair(authority_key);
                        builder.build(&mut rng)
                    })
                    .collect::<Vec<_>>()
            }

            CommitteeConfig::Validators(v) => v,

            CommitteeConfig::AccountKeys(keys) => {
                // See above re fixed protocol keys
                let (_, protocol_keys) = Committee::new_simple_test_committee_of_size(keys.len());
                keys.into_iter()
                    .zip(protocol_keys)
                    .map(|(account_key, protocol_key)| {
                        let mut builder = ValidatorGenesisConfigBuilder::new()
                            .with_protocol_key_pair(protocol_key)
                            .with_account_key_pair(account_key);
                        builder.build(&mut rng)
                    })
                    .collect::<Vec<_>>()
            }
            CommitteeConfig::Deterministic((size, keys)) => {
                // If no keys are provided, generate them.
                let keys = keys.unwrap_or(
                    (0..size.get())
                        .map(|_| SomaKeyPair::Ed25519(get_key_pair_from_rng(&mut rng).1))
                        .collect(),
                );

                let mut configs = vec![];
                for (i, key) in keys.into_iter().enumerate() {
                    let port_offset = 8000 + i * 10;
                    let mut builder = ValidatorGenesisConfigBuilder::new()
                        .with_ip("127.0.0.1".to_owned())
                        .with_account_key_pair(key)
                        .with_deterministic_ports(port_offset as u16);
                    configs.push(builder.build(&mut rng));
                }
                configs
            }
        };

        let genesis_config = self
            .genesis_config
            .unwrap_or_else(GenesisConfig::for_local_testing);

        let account_keys = genesis_config.generate_accounts(&mut rng).unwrap();

        let genesis = Genesis::new(
            VerifiedTransaction::new_genesis_transaction().into_inner(),
            SystemState::create(
                validators
                    .iter()
                    .map(|v| {
                        Validator::new(
                            (&v.account_key_pair.public()).into(),
                            (v.key_pair.public()).clone(),
                            v.network_key_pair.public().into(),
                            v.worker_key_pair.public().into(),
                            v.network_address.clone(),
                            v.consensus_address.clone(),
                            v.network_address.clone(),
                            10000 / validators.len() as u64,
                        )
                    })
                    .collect(),
                0,
                SystemParameters::default(),
            ),
        );

        let validator_configs = validators
            .into_iter()
            .enumerate()
            .map(|(idx, validator)| {
                let mut builder = ValidatorConfigBuilder::new();
                builder.build(validator, genesis.clone())
            })
            .collect();
        NetworkConfig {
            validator_configs,
            genesis,
            account_keys,
        }
    }
}
