use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    ops::Div,
    time::{Duration, Instant, SystemTime},
};

use crate::{
    base::SomaAddress,
    committee::{Committee, CommitteeWithNetworkMetadata},
    config::{node_config::NodeConfig, p2p_config::SeedPeer},
    crypto::{get_key_pair_from_rng, PublicKey, SomaKeyPair},
    digests::TransactionDigest,
    effects::{ExecutionStatus, TransactionEffects},
    genesis::{self, Genesis},
    multiaddr::Multiaddr,
    object::{self, Object, ObjectData, ObjectID, ObjectType, Owner, Version},
    peer_id::PeerId,
    system_state::{
        validator::{self, Validator},
        SystemParameters, SystemState,
    },
    temporary_store::TemporaryStore,
    transaction::{InputObjects, VerifiedTransaction},
    SYSTEM_STATE_OBJECT_ID, SYSTEM_STATE_OBJECT_SHARED_VERSION,
};
use fastcrypto::traits::KeyPair;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::{
    genesis_config::{
        AccountConfig, GenesisConfig, TokenAllocation, TokenDistributionScheduleBuilder,
        ValidatorGenesisConfig, ValidatorGenesisConfigBuilder, DEFAULT_GAS_AMOUNT,
    },
    node_config::ValidatorConfigBuilder,
};

pub enum CommitteeConfig {
    Size(NonZeroUsize),
    Validators(Vec<ValidatorGenesisConfig>),
    AccountKeys(Vec<SomaKeyPair>),
    /// Indicates that a committee should be deterministically generated, using the provided rng
    /// as a source of randomness as well as generating deterministic network port information.
    Deterministic((NonZeroUsize, Option<Vec<SomaKeyPair>>)),
}

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

        let (account_keys, allocations) = genesis_config.generate_accounts(&mut rng).unwrap();

        let token_distribution_schedule = {
            let mut builder = TokenDistributionScheduleBuilder::new();
            for allocation in allocations {
                builder.add_allocation(allocation);
            }
            // Add allocations for each validator
            for validator in &validators {
                let account_key: PublicKey = validator.account_key_pair.public();
                let address = SomaAddress::from(&account_key);
                // Give each validator some gas so they can pay for their transactions.
                let gas_coin = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: DEFAULT_GAS_AMOUNT,
                    staked_with_validator: None,
                };
                let stake = TokenAllocation {
                    recipient_address: address,
                    amount_shannons: validator.stake,
                    staked_with_validator: Some(address),
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
            validators
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
                        0,
                        v.commission_rate,
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

        let tx = VerifiedTransaction::new_genesis_transaction(objects.clone()).into_inner();
        let digest = *tx.digest();

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

        let genesis = Genesis::new(
            tx.clone(),
            effects,
            inner.written.iter().map(|(_, o)| o.clone()).collect(),
        );

        let seed_peers: Vec<SeedPeer> = validators
            .iter()
            .map(|config| SeedPeer {
                peer_id: Some(PeerId(
                    config.network_key_pair.public().into_inner().0.to_bytes(),
                )),
                address: config.p2p_address.clone(),
            })
            .collect();

        let validator_configs = validators
            .into_iter()
            .enumerate()
            .map(|(idx, validator)| {
                let mut builder = ValidatorConfigBuilder::new();
                builder.build(validator, genesis.clone(), seed_peers.clone())
            })
            .collect();
        NetworkConfig {
            validator_configs,
            genesis,
            account_keys,
        }
    }
}
