use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    ops::Div,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use crate::{
    base::AuthorityName,
    base::SomaAddress,
    committee::{Committee, CommitteeWithNetworkMetadata},
    config::{
        node_config::{get_key_path, NodeConfig, ENCODERS_DB_NAME},
        p2p_config::SeedPeer,
    },
    consensus::stake_aggregator::StakeAggregator,
    crypto::{
        get_key_pair_from_rng, AuthorityKeyPair, AuthorityPublicKeyBytes, AuthoritySignInfo,
        AuthorityStrongQuorumSignInfo, NetworkPublicKey, PublicKey, SomaKeyPair,
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
        encoder::Encoder,
        epoch_start::EpochStartSystemStateTrait,
        validator::{self, Validator},
        SystemState, SystemStateTrait,
    },
    temporary_store::TemporaryStore,
    transaction::{
        CertifiedTransaction, InputObjects, SignedTransaction, Transaction, TransactionData,
        VerifiedTransaction,
    },
    SYSTEM_STATE_OBJECT_ID, SYSTEM_STATE_OBJECT_SHARED_VERSION,
};
use crate::{config::Config, shard_crypto::keys::EncoderKeyPair};
use fastcrypto::{bls12381::min_sig::BLS12381KeyPair, traits::KeyPair};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::{
    encoder_config::{EncoderCommitteeConfig, EncoderConfig, EncoderGenesisConfigBuilder},
    genesis_config::{
        AccountConfig, GenesisConfig, TokenAllocation, TokenDistributionScheduleBuilder,
        ValidatorGenesisConfig, ValidatorGenesisConfigBuilder, DEFAULT_GAS_AMOUNT,
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
    Mixed {
        consensus_count: NonZeroUsize,
        networking_count: NonZeroUsize,
    },
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
    pub encoder_configs: Vec<EncoderConfig>,
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
    committee: CommitteeConfig,
    encoder_committee: EncoderCommitteeConfig,
    genesis_config: Option<GenesisConfig>,
    supported_protocol_versions_config: Option<ProtocolVersionsConfig>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            rng: Some(OsRng),
            committee: CommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
            encoder_committee: EncoderCommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
            genesis_config: None,
            supported_protocol_versions_config: None,
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

    pub fn encoder_committee(mut self, committee: EncoderCommitteeConfig) -> Self {
        self.encoder_committee = committee;
        self
    }

    pub fn encoder_committee_size(mut self, committee_size: NonZeroUsize) -> Self {
        self.encoder_committee = EncoderCommitteeConfig::Size(committee_size);
        self
    }

    pub fn rng<N: rand::RngCore + rand::CryptoRng>(self, rng: N) -> ConfigBuilder<N> {
        ConfigBuilder {
            rng: Some(rng),
            committee: self.committee,
            encoder_committee: self.encoder_committee,
            genesis_config: self.genesis_config,
            supported_protocol_versions_config: self.supported_protocol_versions_config,
        }
    }

    pub fn with_current_unix_timestamp_ms(mut self) -> Self {
        let duration_since_unix_epoch = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("SystemTime before UNIX EPOCH!");

        self.get_or_init_genesis_config()
            .parameters
            .chain_start_timestamp_ms = duration_since_unix_epoch.as_millis() as u64;
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
        let encoder_committee = self.encoder_committee;

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
            CommitteeConfig::Mixed {
                consensus_count,
                networking_count,
            } => {
                let mut configs = vec![];

                // Generate consensus validators
                let (_, consensus_keys) =
                    Committee::new_simple_test_committee_of_size(consensus_count.get());
                for (i, authority_key) in consensus_keys.into_iter().enumerate() {
                    // let port_offset = 8000 + i * 10;
                    let builder =
                        ValidatorGenesisConfigBuilder::new().with_protocol_key_pair(authority_key);
                    // .with_ip("127.0.0.1".to_owned())
                    // .with_deterministic_ports(port_offset as u16);
                    configs.push(builder.build(&mut rng));
                }

                // Generate networking validators
                for i in 0..networking_count.get() {
                    let builder = ValidatorGenesisConfigBuilder::new()
                        // .with_ip("127.0.0.1".to_owned())
                        // .with_deterministic_ports(port_offset as u16)
                        .as_networking_only(); // Mark as networking-only
                    configs.push(builder.build(&mut rng));
                }

                configs
            }
        };

        // Generate encoders
        let encoders = match encoder_committee {
            EncoderCommitteeConfig::Size(size) => (0..size.get())
                .map(|_| {
                    let mut builder = EncoderGenesisConfigBuilder::new();
                    builder.build(&mut rng)
                })
                .collect::<Vec<_>>(),
            EncoderCommitteeConfig::Encoders(e) => e,
            EncoderCommitteeConfig::EncoderKeys(keys) => keys
                .into_iter()
                .map(|encoder_key| {
                    let mut builder =
                        EncoderGenesisConfigBuilder::new().with_encoder_key_pair(encoder_key);
                    builder.build(&mut rng)
                })
                .collect::<Vec<_>>(),
            EncoderCommitteeConfig::Deterministic((size, keys)) => {
                // If no keys are provided, generate them
                let keys = keys.unwrap_or(
                    (0..size.get())
                        .map(|_| EncoderKeyPair::new(get_key_pair_from_rng(&mut rng).1))
                        .collect(),
                );

                let mut configs = vec![];
                for (i, key) in keys.into_iter().enumerate() {
                    let port_offset = 9000 + i * 10; // Different port range than validators
                    let mut builder = EncoderGenesisConfigBuilder::new()
                        .with_ip("127.0.0.1".to_owned())
                        .with_encoder_key_pair(key)
                        .with_deterministic_ports(port_offset as u16);
                    configs.push(builder.build(&mut rng));
                }
                configs
            }
        };

        let mut genesis_config = self
            .genesis_config
            .unwrap_or_else(GenesisConfig::for_local_testing);

        if genesis_config.parameters.chain_start_timestamp_ms == 0 {
            genesis_config.parameters.chain_start_timestamp_ms = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("SystemTime before UNIX EPOCH!")
                .as_millis()
                as u64;
        }

        let (account_keys, allocations) = genesis_config.generate_accounts(&mut rng).unwrap();

        let (networking_configs, consensus_configs): (Vec<_>, Vec<_>) = all_validators
            .into_iter()
            .partition(|v| v.is_networking_only);

        // Calculate stake to ensure voting power stays below threshold
        let total_consensus_stake: u64 = consensus_configs.iter().map(|v| v.stake).sum();
        let networking_stake = {
            // We want: (stake * 10000) / total_stake < 12
            // So: stake < (12 * total_stake) / 10000
            let max_stake = (11 * (total_consensus_stake + 1)) / 10000; // Use 11 for safety margin
            max_stake
        };

        let token_distribution_schedule = {
            let mut builder = TokenDistributionScheduleBuilder::new();
            for allocation in allocations {
                builder.add_allocation(allocation);
            }
            // Add allocations for each validator
            for validator in &consensus_configs {
                let account_key: PublicKey = validator.account_key_pair.public();
                let address = SomaAddress::from(&account_key);
                // Give each validator some gas so they can pay for their transactions.
                let gas_coin = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: DEFAULT_GAS_AMOUNT * 10,
                    staked_with_validator: None,
                    staked_with_encoder: None,
                };
                let stake = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: validator.stake,
                    staked_with_validator: Some(address),
                    staked_with_encoder: None,
                };
                builder.add_allocation(gas_coin);
                builder.add_allocation(stake);
            }

            for validator in &networking_configs {
                let account_key: PublicKey = validator.account_key_pair.public();
                let address = SomaAddress::from(&account_key);
                // Give each validator some gas so they can pay for their transactions.
                let gas_coin = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: DEFAULT_GAS_AMOUNT * 10,
                    staked_with_validator: None,
                    staked_with_encoder: None,
                };

                let stake = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: std::cmp::min(networking_stake, validator.stake / 100),
                    staked_with_validator: Some(address),
                    staked_with_encoder: None,
                };
                builder.add_allocation(gas_coin);
                builder.add_allocation(stake);
            }

            // Add allocations for each encoder
            for encoder in &encoders {
                let account_key: PublicKey = encoder.account_key_pair.public();
                let address = SomaAddress::from(&account_key);
                // Give each encoder some gas
                let gas_coin = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: DEFAULT_GAS_AMOUNT * 10,
                    staked_with_validator: None,
                    staked_with_encoder: None,
                };
                // Initial encoder stake
                let stake = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: encoder.stake,
                    staked_with_validator: None,
                    staked_with_encoder: Some(address),
                };
                builder.add_allocation(gas_coin);
                builder.add_allocation(stake);
            }

            builder.build()
        };

        let consensus_keypairs: Vec<AuthorityKeyPair> = consensus_configs
            .clone()
            .iter()
            .map(|v| v.key_pair.copy()) // Use copy() method from KeyPair trait
            .collect();

        let all_validators: Vec<ValidatorGenesisConfig> = networking_configs
            .clone()
            .into_iter()
            .chain(consensus_configs.clone().into_iter())
            .collect();

        // Use GenesisBuilder
        let mut genesis_builder = GenesisBuilder::new()
            .with_parameters(genesis_config.clone())
            .with_validators(consensus_configs.clone())
            .with_networking_validators(networking_configs.clone())
            .with_encoders(encoders.clone())
            .with_token_distribution_schedule(token_distribution_schedule);

        // Add validator signatures
        for keypair in &consensus_keypairs {
            genesis_builder = genesis_builder.add_validator_signature(keypair);
        }

        // Build the genesis
        let genesis = genesis_builder.build();

        let seed_peers: Vec<SeedPeer> = all_validators
            .iter()
            .map(|config| SeedPeer {
                peer_id: Some(PeerId(
                    config.network_key_pair.public().into_inner().0.to_bytes(),
                )),
                address: config.p2p_address.clone(),
            })
            .collect();

        let validator_sync_address = all_validators[0].encoder_validator_address.clone(); // TODO: Temporarily using first validator as RPC address
        let validator_sync_network_key = all_validators[0].network_key_pair.public().clone();
        let rpc_address = all_validators[0].rpc_address.clone();

        let validator_configs = all_validators
            .into_iter()
            .enumerate()
            .map(|(idx, validator)| {
                let mut builder = ValidatorConfigBuilder::new();

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

        // let key_path = get_key_path(&account_keypair);
        let encoder_config_directory = tempfile::tempdir().unwrap().into_path();
        let db_path = encoder_config_directory.join(ENCODERS_DB_NAME);

        let encoder_configs = encoders
            .into_iter()
            .map(|encoder| {
                EncoderConfig::new(
                    encoder.account_key_pair,
                    encoder.encoder_key_pair.clone(),
                    encoder.network_key_pair,
                    encoder.internal_network_address,
                    encoder.external_network_address,
                    encoder.object_address,
                    encoder.inference_address,
                    encoder.evaluation_address,
                    rpc_address
                        .to_socket_addr()
                        .expect("Could not turn rpc address into socket address"),
                    PathBuf::from("/project/root"), // Default path, should be configurable
                    PathBuf::from("/entry/point.py"), // Default path, should be configurable
                    validator_sync_address.clone(),
                    validator_sync_network_key.clone(),
                    genesis.clone(),
                    db_path
                        .clone()
                        .join(get_key_path(encoder.encoder_key_pair.inner())),
                )
                .with_epoch_duration(genesis_config.parameters.epoch_duration_ms)
            })
            .collect();

        NetworkConfig {
            validator_configs,
            encoder_configs,
            genesis,
            account_keys,
        }
    }
}
