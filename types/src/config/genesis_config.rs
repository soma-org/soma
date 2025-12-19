use std::net::{IpAddr, SocketAddr};

use super::local_ip_utils;
use crate::{
    base::SomaAddress,
    config::Config,
    crypto::{
        get_key_pair_from_rng, AuthorityKeyPair, NetworkKeyPair, ProtocolKeyPair, SomaKeyPair,
    },
    multiaddr::Multiaddr,
};
use anyhow::Result;
use fastcrypto::traits::KeyPair as _;
use protocol_config::ProtocolVersion;
use serde::{Deserialize, Serialize};
use tracing::info;

// All information needed to build a NodeConfig for a validator.
#[derive(Serialize, Deserialize, Debug)]
pub struct ValidatorGenesisConfig {
    pub key_pair: AuthorityKeyPair,
    pub worker_key_pair: NetworkKeyPair,
    pub account_key_pair: SomaKeyPair,
    pub network_key_pair: NetworkKeyPair,
    pub network_address: Multiaddr,
    pub consensus_address: Multiaddr,
    pub p2p_address: Multiaddr,
    pub encoder_validator_address: Multiaddr,
    pub rpc_address: Multiaddr,
    #[serde(default = "default_stake")]
    pub stake: u64,
    pub commission_rate: u64,
    pub is_networking_only: bool,
}

impl Clone for ValidatorGenesisConfig {
    fn clone(&self) -> Self {
        Self {
            key_pair: self.key_pair.copy(),
            worker_key_pair: self.worker_key_pair.clone(),
            account_key_pair: self.account_key_pair.copy(),
            network_key_pair: self.network_key_pair.clone(),
            network_address: self.network_address.clone(),
            consensus_address: self.consensus_address.clone(),
            p2p_address: self.p2p_address.clone(),
            encoder_validator_address: self.encoder_validator_address.clone(),
            rpc_address: self.rpc_address.clone(),
            stake: self.stake.clone(),
            commission_rate: self.commission_rate.clone(),
            is_networking_only: self.is_networking_only.clone(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AccountConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<SomaAddress>,
    pub gas_amounts: Vec<u64>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct GenesisConfig {
    pub validator_config_info: Option<Vec<ValidatorGenesisConfig>>,
    pub networking_validator_config_info: Option<Vec<ValidatorGenesisConfig>>,
    pub parameters: GenesisCeremonyParameters,
    pub accounts: Vec<AccountConfig>,
}

pub const DEFAULT_GAS_AMOUNT: u64 = 30_000_000_000_000_000;
const DEFAULT_COMMISSION_RATE: u64 = 200;
const DEFAULT_NUMBER_OF_ACCOUNTS: usize = 5;
const DEFAULT_NUMBER_OF_OBJECT_PER_ACCOUNT: usize = 5;

/// The number of Shannons per Soma token
pub const SHANNONS_PER_SOMA: u64 = 1_000_000_000;

/// Total supply denominated in Soma
pub const TOTAL_SUPPLY_SOMA: u64 = 10_000_000_000;

// Note: cannot use checked arithmetic here since `const unwrap` is still unstable.
/// Total supply denominated in Shannons
pub const TOTAL_SUPPLY_SHANNONS: u64 = TOTAL_SUPPLY_SOMA * SHANNONS_PER_SOMA;

impl GenesisConfig {
    pub fn for_local_testing() -> Self {
        Self::custom_genesis(
            DEFAULT_NUMBER_OF_ACCOUNTS,
            DEFAULT_NUMBER_OF_OBJECT_PER_ACCOUNT,
        )
    }

    pub fn for_local_testing_with_addresses(addresses: Vec<SomaAddress>) -> Self {
        Self::custom_genesis_with_addresses(addresses, DEFAULT_NUMBER_OF_OBJECT_PER_ACCOUNT)
    }

    pub fn custom_genesis(num_accounts: usize, num_objects_per_account: usize) -> Self {
        let mut accounts = Vec::new();
        for _ in 0..num_accounts {
            accounts.push(AccountConfig {
                address: None,
                gas_amounts: vec![DEFAULT_GAS_AMOUNT * 10; num_objects_per_account],
            })
        }

        Self {
            accounts,
            ..Default::default()
        }
    }

    pub fn custom_genesis_with_addresses(
        addresses: Vec<SomaAddress>,
        num_objects_per_account: usize,
    ) -> Self {
        let mut accounts = Vec::new();
        for address in addresses {
            accounts.push(AccountConfig {
                address: Some(address),
                gas_amounts: vec![DEFAULT_GAS_AMOUNT; num_objects_per_account],
            })
        }

        Self {
            accounts,
            ..Default::default()
        }
    }

    pub fn generate_accounts<R: rand::RngCore + rand::CryptoRng>(
        &self,
        mut rng: R,
    ) -> Result<(Vec<SomaKeyPair>, Vec<TokenAllocation>)> {
        let mut addresses = Vec::new();
        let mut allocations = Vec::new();

        info!("Creating accounts and token allocations...");

        let mut keys = Vec::new();
        for account in &self.accounts {
            let address = if let Some(address) = account.address {
                address
            } else {
                let (address, keypair) = get_key_pair_from_rng(&mut rng);
                keys.push(SomaKeyPair::Ed25519(keypair));
                address
            };

            addresses.push(address);

            // Populate gas itemized objects
            account.gas_amounts.iter().for_each(|a| {
                allocations.push(TokenAllocation {
                    recipient_address: address,
                    amount_shannons: *a,
                    staked_with_validator: None,
                    staked_with_encoder: None,
                });
            });
        }

        Ok((keys, allocations))
    }
}

impl Config for GenesisConfig {}

#[derive(Default)]
pub struct ValidatorGenesisConfigBuilder {
    protocol_key_pair: Option<AuthorityKeyPair>,
    account_key_pair: Option<SomaKeyPair>,
    ip: Option<String>,
    is_networking_only: Option<bool>,
    /// If set, the validator will use deterministic addresses based on the port offset.
    /// This is useful for benchmarking.
    port_offset: Option<u16>,
    stake: Option<u64>,
}

impl ValidatorGenesisConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_protocol_key_pair(mut self, key_pair: AuthorityKeyPair) -> Self {
        self.protocol_key_pair = Some(key_pair);
        self
    }

    pub fn with_account_key_pair(mut self, key_pair: SomaKeyPair) -> Self {
        self.account_key_pair = Some(key_pair);
        self
    }

    pub fn with_ip(mut self, ip: String) -> Self {
        self.ip = Some(ip);
        self
    }

    pub fn as_networking_only(mut self) -> Self {
        self.is_networking_only = Some(true);
        self
    }

    pub fn with_stake(mut self, stake: u64) -> Self {
        self.stake = Some(stake);
        self
    }

    pub fn with_deterministic_ports(mut self, port_offset: u16) -> Self {
        self.port_offset = Some(port_offset);
        self
    }

    pub fn build<R: rand::RngCore + rand::CryptoRng>(self, rng: &mut R) -> ValidatorGenesisConfig {
        let ip = self.ip.unwrap_or_else(local_ip_utils::get_new_ip);
        let stake = self.stake.unwrap_or(default_stake());
        let localhost = local_ip_utils::localhost_for_testing();
        let protocol_key_pair = self
            .protocol_key_pair
            .unwrap_or_else(|| get_key_pair_from_rng(rng).1);
        let account_key_pair = self
            .account_key_pair
            .unwrap_or_else(|| SomaKeyPair::Ed25519(get_key_pair_from_rng(rng).1));

        let (worker_key_pair, network_key_pair): (NetworkKeyPair, NetworkKeyPair) = (
            NetworkKeyPair::new(get_key_pair_from_rng(rng).1),
            NetworkKeyPair::new(get_key_pair_from_rng(rng).1),
        );

        let (
            network_address,
            consensus_address,
            p2p_address,
            encoder_validator_address,
            rpc_address,
        ) = if let Some(offset) = self.port_offset {
            (
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 1),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 2),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 3),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 4),
            )
        } else {
            (
                local_ip_utils::new_tcp_address_for_testing(&ip),
                local_ip_utils::new_tcp_address_for_testing(&ip),
                local_ip_utils::new_tcp_address_for_testing(&ip),
                local_ip_utils::new_tcp_address_for_testing(&ip),
                local_ip_utils::new_tcp_address_for_testing(&ip),
            )
        };

        ValidatorGenesisConfig {
            key_pair: protocol_key_pair,
            worker_key_pair,
            account_key_pair: account_key_pair.into(),
            network_key_pair,
            network_address,
            consensus_address,
            p2p_address,
            encoder_validator_address,
            rpc_address,
            stake,
            commission_rate: DEFAULT_COMMISSION_RATE,
            is_networking_only: self.is_networking_only.unwrap_or(false),
        }
    }
}

/// Initial set of parameters for a chain.
#[derive(Serialize, Deserialize, Clone)]
pub struct GenesisCeremonyParameters {
    #[serde(default = "GenesisCeremonyParameters::default_timestamp_ms")]
    pub chain_start_timestamp_ms: u64,

    /// protocol version that the chain starts at.
    #[serde(default = "ProtocolVersion::max")]
    pub protocol_version: ProtocolVersion,

    /// The duration of an epoch, in milliseconds.
    #[serde(default = "GenesisCeremonyParameters::default_epoch_duration_ms")]
    pub epoch_duration_ms: u64,

    /// The amount of rewards to be drawn down per distribution.
    #[serde(default = "GenesisCeremonyParameters::default_emission_per_epoch")]
    pub emission_per_epoch: u64,

    /// Seed target embeddings (if empty, random embeddings will be generated)
    #[serde(default)] // TODO: define seed target embeddings
    pub seed_target_embeddings: Vec<Vec<u8>>,
}

impl GenesisCeremonyParameters {
    pub fn new() -> Self {
        Self {
            chain_start_timestamp_ms: Self::default_timestamp_ms(),
            protocol_version: ProtocolVersion::max(),
            epoch_duration_ms: Self::default_epoch_duration_ms(),
            emission_per_epoch: Self::default_emission_per_epoch(),
            seed_target_embeddings: vec![],
        }
    }

    fn default_timestamp_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    fn default_epoch_duration_ms() -> u64 {
        // 24 hrs
        24 * 60 * 60 * 1000
    }

    fn default_emission_per_epoch() -> u64 {
        // 1M SOMA
        1_000_000 * SHANNONS_PER_SOMA // TODO: set the emission slope here!
    }
}

impl Default for GenesisCeremonyParameters {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TokenAllocation {
    pub recipient_address: SomaAddress,
    pub amount_shannons: u64,

    /// Indicates if this allocation should be staked at genesis and with which validator
    pub staked_with_validator: Option<SomaAddress>,
    pub staked_with_encoder: Option<SomaAddress>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TokenDistributionSchedule {
    pub emission_fund_shannons: u64,
    pub allocations: Vec<TokenAllocation>,
}

impl TokenDistributionSchedule {
    pub fn validate(&self) {
        let mut total = self.emission_fund_shannons;

        for allocation in &self.allocations {
            total += allocation.amount_shannons;
        }

        if total != TOTAL_SUPPLY_SHANNONS {
            panic!(
                "TokenDistributionSchedule adds up to {total} and not expected \
                 {TOTAL_SUPPLY_SHANNONS}"
            );
        }
    }

    pub fn check_all_stake_operations_are_for_valid_validators<
        I: IntoIterator<Item = SomaAddress>,
    >(
        &self,
        validators: I,
    ) {
        use std::collections::HashMap;

        let mut validators: HashMap<SomaAddress, u64> =
            validators.into_iter().map(|a| (a, 0)).collect();

        // Check that all allocations are for valid validators, while summing up all allocations
        // for each validator
        for allocation in &self.allocations {
            if let Some(staked_with_validator) = &allocation.staked_with_validator {
                *validators
                    .get_mut(staked_with_validator)
                    .expect("allocation must be staked with valid validator") +=
                    allocation.amount_shannons;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenDistributionScheduleBuilder {
    pool: u64,
    allocations: Vec<TokenAllocation>,
}

impl TokenDistributionScheduleBuilder {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            pool: TOTAL_SUPPLY_SHANNONS,
            allocations: vec![],
        }
    }

    pub fn add_allocation(&mut self, allocation: TokenAllocation) {
        self.pool = self.pool.checked_sub(allocation.amount_shannons).unwrap();
        self.allocations.push(allocation);
    }

    pub fn build(&self) -> TokenDistributionSchedule {
        let schedule = TokenDistributionSchedule {
            emission_fund_shannons: self.pool,
            allocations: self.allocations.clone(),
        };

        schedule.validate();
        schedule
    }
}

fn default_stake() -> u64 {
    20_000_000_000_000_000
}
