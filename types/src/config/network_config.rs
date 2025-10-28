use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    ops::Div,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use crate::{
    base::SomaAddress,
    committee::{Committee, CommitteeWithNetworkMetadata},
    config::{
        node_config::{get_key_path, NodeConfig, ENCODERS_DB_NAME},
        p2p_config::SeedPeer,
    },
    consensus::stake_aggregator::StakeAggregator,
    crypto::{
        get_key_pair_from_rng, AuthorityPublicKeyBytes, AuthoritySignInfo,
        AuthorityStrongQuorumSignInfo, NetworkPublicKey, PublicKey, SomaKeyPair,
    },
    digests::TransactionDigest,
    effects::{ExecutionStatus, TransactionEffects},
    error::{SomaError, SomaResult},
    genesis::{self, Genesis},
    intent::Intent,
    multiaddr::Multiaddr,
    object::{self, Object, ObjectData, ObjectID, ObjectType, Owner, Version},
    peer_id::PeerId,
    system_state::{
        encoder::Encoder,
        epoch_start::EpochStartSystemStateTrait,
        validator::{self, Validator},
        SystemParameters, SystemState, SystemStateTrait,
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
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            rng: Some(OsRng),
            committee: CommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
            encoder_committee: EncoderCommitteeConfig::Size(NonZeroUsize::new(1).unwrap()),
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
        }
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

        let genesis_config = self
            .genesis_config
            .unwrap_or_else(GenesisConfig::for_local_testing);

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

        let now = Instant::now();
        let duration_since_unix_epoch =
            match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
                Ok(d) => d,
                Err(e) => panic!("SystemTime before UNIX EPOCH! {e}"),
            };
        // let unix_epoch_instant = now.checked_sub(duration_since_unix_epoch).unwrap();

        let mut system_state = SystemState::create(
            consensus_configs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    Validator::new(
                        (&v.account_key_pair.public()).into(),
                        (v.key_pair.public()).clone(),
                        v.network_key_pair.public().into(),
                        v.worker_key_pair.public().into(),
                        v.network_address.clone(),
                        v.consensus_address.clone(),
                        v.network_address.clone(),
                        v.encoder_validator_address.clone(),
                        0,
                        v.commission_rate,
                        ObjectID::random(),
                    )
                })
                .collect(),
            networking_configs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    Validator::new(
                        (&v.account_key_pair.public()).into(),
                        (v.key_pair.public()).clone(),
                        v.network_key_pair.public().into(),
                        v.worker_key_pair.public().into(),
                        v.network_address.clone(),
                        v.consensus_address.clone(),
                        v.network_address.clone(),
                        v.encoder_validator_address.clone(),
                        0,
                        v.commission_rate,
                        ObjectID::random(),
                    )
                })
                .collect(),
            encoders
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    Encoder::new(
                        (&e.account_key_pair.public()).into(),
                        e.encoder_key_pair.public().clone(),
                        e.network_key_pair.public().clone(),
                        e.internal_network_address.clone(),
                        e.external_network_address.clone(),
                        e.object_address.clone(),
                        0,
                        e.commission_rate,
                        e.byte_price,
                        ObjectID::random(),
                    )
                })
                .collect(),
            duration_since_unix_epoch.as_millis() as u64,
            SystemParameters {
                epoch_duration_ms: genesis_config.parameters.epoch_duration_ms,
                ..Default::default()
            },
            token_distribution_schedule.stake_subsidy_fund_shannons,
            genesis_config
                .parameters
                .stake_subsidy_initial_distribution_amount,
            genesis_config.parameters.stake_subsidy_period_length,
            genesis_config.parameters.stake_subsidy_decrease_rate,
        );

        let mut objects = vec![];

        for allocation in token_distribution_schedule.allocations {
            if let Some(validator) = allocation.staked_with_validator {
                let staked_soma = system_state
                    .request_add_stake_at_genesis(
                        allocation.recipient_address,
                        validator,
                        allocation.amount_shannons,
                    )
                    .expect("Could not stake in validator at Genesis.");
                let staked_soma_object = Object::new_staked_soma_object(
                    ObjectID::random(),
                    staked_soma,
                    Owner::AddressOwner(allocation.recipient_address),
                    TransactionDigest::default(),
                );
                objects.push(staked_soma_object);
            } else if let Some(encoder) = allocation.staked_with_encoder {
                let staked_soma = system_state
                    .request_add_encoder_stake_at_genesis(
                        allocation.recipient_address,
                        encoder,
                        allocation.amount_shannons,
                    )
                    .expect("Could not stake in encoder at Genesis.");
                let staked_soma_object = Object::new_staked_soma_object(
                    ObjectID::random(),
                    staked_soma,
                    Owner::AddressOwner(allocation.recipient_address),
                    TransactionDigest::default(),
                );
                objects.push(staked_soma_object);
            } else {
                let coin_object = Object::new_coin(
                    ObjectID::random(),
                    allocation.amount_shannons,
                    Owner::AddressOwner(allocation.recipient_address),
                    TransactionDigest::default(),
                );
                objects.push(coin_object);
            }
        }

        system_state.validators.set_voting_power();
        system_state.encoders.set_voting_power();

        // Initialize current epoch committees
        let current_committees = system_state.build_committees_for_epoch(0);
        system_state.committees[1] = Some(current_committees);

        let state_object = Object::new(
            ObjectData::new_with_id(
                SYSTEM_STATE_OBJECT_ID,
                ObjectType::SystemState,
                Version::MIN,
                bcs::to_bytes(&system_state).unwrap(),
            ),
            Owner::Shared {
                initial_shared_version: Version::new(),
            },
            TransactionDigest::default(),
        );

        objects.push(state_object);

        let committee = system_state.into_epoch_start_state().get_committee();
        let unsigned_tx =
            VerifiedTransaction::new_genesis_transaction(objects.clone()).into_inner();

        // Collect all signatures directly
        let mut signatures = Vec::new();
        for validator in &consensus_configs {
            let authority_name = AuthorityPublicKeyBytes::from(validator.key_pair.public());

            let sig_info = AuthoritySignInfo::new(
                committee.epoch(),
                unsigned_tx.data(),
                Intent::soma_transaction(),
                authority_name,
                &validator.key_pair,
            );
            signatures.push(sig_info);
        }

        // Create the quorum signature directly
        let cert_sig =
            AuthorityStrongQuorumSignInfo::new_from_auth_sign_infos(signatures, &committee)
                .unwrap();

        let certified_tx =
            CertifiedTransaction::new_from_data_and_sig(unsigned_tx.into_data(), cert_sig);

        // Verify the certificate
        certified_tx
            .verify_committee_sigs_only(&committee)
            .expect("Genesis certificate should verify");

        let digest = *certified_tx.digest();

        // Create the input objects map for TemporaryStore
        let input_objects = InputObjects::new(Vec::new());
        let shared_object_refs = Vec::new(); // No shared objects in the input
        let receiving_objects = Vec::new(); // No receiving objects

        // Create a TemporaryStore for the genesis transaction
        let mut temp_store = TemporaryStore::new(
            input_objects,
            receiving_objects,
            digest,
            0, // epoch_id
        );

        // Add the created objects to the store
        for object in objects {
            temp_store.create_object(object);
        }

        // Generate effects using the into_effects method
        let (inner, effects) = temp_store.into_effects(
            shared_object_refs,
            &digest,
            BTreeSet::new(), // No transaction dependencies for genesis
            ExecutionStatus::Success,
            0, // epoch_id
            None,
        );

        let genesis = Genesis::new_with_certified_tx(
            certified_tx.clone(),
            effects,
            inner.written.iter().map(|(_, o)| o.clone()).collect(),
        );

        let all_validators: Vec<ValidatorGenesisConfig> = networking_configs
            .into_iter()
            .chain(consensus_configs.into_iter())
            .collect();

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
        let rpc_address = all_validators[0].rpc_address.clone();

        let validator_configs = all_validators
            .into_iter()
            .enumerate()
            .map(|(idx, validator)| {
                let mut builder = ValidatorConfigBuilder::new();
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
                    encoder.local_object_address,
                    encoder.evaluation_address,
                    rpc_address
                        .to_socket_addr()
                        .expect("Could not turn rpc address into socket address"),
                    PathBuf::from("/project/root"), // Default path, should be configurable
                    PathBuf::from("/entry/point.py"), // Default path, should be configurable
                    validator_sync_address.clone(),
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
