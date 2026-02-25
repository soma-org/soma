// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    ops::Div,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use crate::config::Config;
use crate::{
    SYSTEM_STATE_OBJECT_ID, SYSTEM_STATE_OBJECT_SHARED_VERSION,
    base::AuthorityName,
    base::SomaAddress,
    committee::{Committee, CommitteeWithNetworkMetadata},
    config::{
        node_config::{NodeConfig, get_key_path},
        p2p_config::SeedPeer,
    },
    consensus::stake_aggregator::StakeAggregator,
    crypto::{
        AuthorityKeyPair, AuthorityPublicKeyBytes, AuthoritySignInfo,
        AuthorityStrongQuorumSignInfo, NetworkPublicKey, PublicKey, SomaKeyPair,
        get_key_pair_from_rng,
    },
    digests::TransactionDigest,
    effects::{ExecutionStatus, TransactionEffects},
    error::{SomaError, SomaResult},
    genesis::{self, Genesis},
    genesis_builder::GenesisBuilder,
    intent::Intent,
    multiaddr::Multiaddr,
    object::{self, Object, ObjectData, ObjectID, ObjectType, Owner, Version},
    peer_id::PeerId,
    supported_protocol_versions::SupportedProtocolVersions,
    system_state::{
        SystemState, SystemStateTrait,
        epoch_start::EpochStartSystemStateTrait,
        validator::{self, Validator},
    },
    temporary_store::TemporaryStore,
    transaction::{
        CertifiedTransaction, InputObjects, SignedTransaction, Transaction, TransactionData,
        VerifiedTransaction,
    },
};
use fastcrypto::{bls12381::min_sig::BLS12381KeyPair, traits::KeyPair};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use tempfile::tempfile;
use tracing::info;

use super::{
    genesis_config::{
        AccountConfig, DEFAULT_GAS_AMOUNT, GenesisConfig, TokenAllocation,
        TokenDistributionScheduleBuilder, ValidatorGenesisConfig, ValidatorGenesisConfigBuilder,
    },
    node_config::ValidatorConfigBuilder,
};

#[derive(Debug)]
pub enum CommitteeConfig {
    Size(NonZeroUsize),
    Validators(Vec<ValidatorGenesisConfig>),
    AccountKeys(Vec<SomaKeyPair>),
    /// Indicates that a committee should be deterministically generated, using the provided rng
    /// as a source of randomness as well as generating deterministic network port information.
    Deterministic((NonZeroUsize, Option<Vec<SomaKeyPair>>)),
}

pub type SupportedProtocolVersionsCallback = Arc<
    dyn Fn(
            usize,                 /* validator idx */
            Option<AuthorityName>, /* None for fullnode */
        ) -> SupportedProtocolVersions
        + Send
        + Sync
        + 'static,
>;

#[derive(Clone)]
pub enum ProtocolVersionsConfig {
    // use SYSTEM_DEFAULT
    Default,
    // Use one range for all validators.
    Global(SupportedProtocolVersions),
    // A closure that returns the versions for each validator.
    // TODO: This doesn't apply to fullnodes.
    PerValidator(SupportedProtocolVersionsCallback),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NetworkConfig {
    pub validator_configs: Vec<NodeConfig>,
    pub account_keys: Vec<SomaKeyPair>,
    pub genesis: genesis::Genesis,
}

impl Config for NetworkConfig {}

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
    config_directory: PathBuf,
    committee: CommitteeConfig,
    genesis_config: Option<GenesisConfig>,
    supported_protocol_versions_config: Option<ProtocolVersionsConfig>,
}

impl ConfigBuilder {
    pub fn new<P: AsRef<Path>>(config_directory: P) -> Self {
        Self {
            rng: Some(OsRng),
            config_directory: config_directory.as_ref().into(),
            committee: CommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),

            genesis_config: None,
            supported_protocol_versions_config: None,
        }
    }

    pub fn new_with_temp_dir() -> Self {
        Self::new(tempfile::tempdir().unwrap().keep())
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

    pub fn with_validators(mut self, validators: Vec<ValidatorGenesisConfig>) -> Self {
        self.committee = CommitteeConfig::Validators(validators);
        self
    }

    pub fn rng<N: rand::RngCore + rand::CryptoRng>(self, rng: N) -> ConfigBuilder<N> {
        ConfigBuilder {
            rng: Some(rng),
            config_directory: self.config_directory,
            committee: self.committee,

            genesis_config: self.genesis_config,
            supported_protocol_versions_config: self.supported_protocol_versions_config,
        }
    }

    pub fn with_current_unix_timestamp_ms(mut self) -> Self {
        let duration_since_unix_epoch = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("SystemTime before UNIX EPOCH!");

        self.get_or_init_genesis_config().parameters.chain_start_timestamp_ms =
            duration_since_unix_epoch.as_millis() as u64;
        self
    }

    pub fn with_supported_protocol_versions(mut self, c: SupportedProtocolVersions) -> Self {
        self.supported_protocol_versions_config = Some(ProtocolVersionsConfig::Global(c));
        self
    }

    pub fn with_supported_protocol_version_callback(
        mut self,
        func: SupportedProtocolVersionsCallback,
    ) -> Self {
        self.supported_protocol_versions_config = Some(ProtocolVersionsConfig::PerValidator(func));
        self
    }

    pub fn with_supported_protocol_versions_config(mut self, c: ProtocolVersionsConfig) -> Self {
        self.supported_protocol_versions_config = Some(c);
        self
    }

    pub fn with_accounts(mut self, accounts: Vec<AccountConfig>) -> Self {
        self.get_or_init_genesis_config().accounts = accounts;
        self
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
        let all_validators = match committee {
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

        let mut genesis_config =
            self.genesis_config.unwrap_or_else(GenesisConfig::for_local_testing);

        if genesis_config.parameters.chain_start_timestamp_ms == 0 {
            genesis_config.parameters.chain_start_timestamp_ms = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("SystemTime before UNIX EPOCH!")
                .as_millis()
                as u64;
        }

        let (account_keys, allocations) = genesis_config.generate_accounts(&mut rng).unwrap();

        let token_distribution_schedule = {
            let mut builder = TokenDistributionScheduleBuilder::new();
            for allocation in allocations {
                builder.add_allocation(allocation);
            }
            // Add allocations for each validator
            for validator in &all_validators {
                let account_key: PublicKey = validator.account_key_pair.public();
                let address = SomaAddress::from(&account_key);
                // Give each validator some gas so they can pay for their transactions.
                let gas_coin = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: DEFAULT_GAS_AMOUNT * 10,
                    staked_with_validator: None,
                    staked_with_model: None,
                };
                let stake = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: validator.stake,
                    staked_with_validator: Some(address),
                    staked_with_model: None,
                };
                builder.add_allocation(gas_coin);
                builder.add_allocation(stake);
            }

            // Add allocations for genesis model stakes
            for model in &genesis_config.genesis_models {
                if model.initial_stake > 0 {
                    let stake = TokenAllocation {
                        recipient_address: model.owner,
                        amount_shannons: model.initial_stake,
                        staked_with_validator: None,
                        staked_with_model: Some(model.model_id),
                    };
                    builder.add_allocation(stake);
                }
            }

            builder.build()
        };

        // Use GenesisBuilder
        let mut genesis_builder = GenesisBuilder::new()
            .with_parameters(genesis_config.parameters.clone())
            .with_validator_configs(all_validators.clone())
            .with_token_distribution_schedule(token_distribution_schedule)
            .with_genesis_models(genesis_config.genesis_models);

        let consensus_keypairs: Vec<AuthorityKeyPair> = all_validators
            .clone()
            .iter()
            .map(|v| v.key_pair.copy()) // Use copy() method from KeyPair trait
            .collect();
        // Add validator signatures
        for keypair in &consensus_keypairs {
            genesis_builder = genesis_builder.add_validator_signature(keypair);
        }

        // Build the genesis
        let genesis = genesis_builder.build();

        let seed_peers: Vec<SeedPeer> = all_validators
            .iter()
            .map(|config| SeedPeer {
                peer_id: Some(PeerId(config.network_key_pair.public().into_inner().0.to_bytes())),
                address: config.p2p_address.clone(),
            })
            .collect();

        let rpc_address = all_validators[0].rpc_address.clone(); // TODO: Temporarily using first validator as RPC address

        let validator_configs = all_validators
            .into_iter()
            .enumerate()
            .map(|(idx, validator)| {
                let mut builder = ValidatorConfigBuilder::new()
                    .with_config_directory(self.config_directory.clone());

                if let Some(spvc) = &self.supported_protocol_versions_config {
                    let supported_versions = match spvc {
                        ProtocolVersionsConfig::Default => {
                            SupportedProtocolVersions::SYSTEM_DEFAULT
                        }
                        ProtocolVersionsConfig::Global(v) => *v,
                        ProtocolVersionsConfig::PerValidator(func) => {
                            func(idx, Some(validator.key_pair.public().into()))
                        }
                    };
                    builder = builder.with_supported_protocol_versions(supported_versions);
                }

                builder.build(validator, genesis.clone(), seed_peers.clone())
            })
            .collect();

        NetworkConfig { validator_configs, genesis, account_keys }
    }
}
