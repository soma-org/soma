use super::local_ip_utils;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;
use types::{
    base::SomaAddress,
    crypto::{
        get_key_pair_from_rng, AuthorityKeyPair, NetworkKeyPair, ProtocolKeyPair, SomaKeyPair,
    },
    multiaddr::Multiaddr,
};

// All information needed to build a NodeConfig for a validator.
#[derive(Serialize, Deserialize)]
pub struct ValidatorGenesisConfig {
    pub key_pair: AuthorityKeyPair,
    pub worker_key_pair: NetworkKeyPair,
    pub account_key_pair: SomaKeyPair,
    pub network_key_pair: NetworkKeyPair,
    pub network_address: Multiaddr,
    pub consensus_address: Multiaddr,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AccountConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<SomaAddress>,
}

// All information needed to build a NodeConfig for a state sync fullnode.
#[derive(Serialize, Deserialize, Debug)]
pub struct SsfnGenesisConfig {
    pub p2p_address: Multiaddr,
    pub network_key_pair: Option<NetworkKeyPair>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct GenesisConfig {
    pub ssfn_config_info: Option<Vec<SsfnGenesisConfig>>,
    pub validator_config_info: Option<Vec<ValidatorGenesisConfig>>,
    pub accounts: Vec<AccountConfig>,
}

const DEFAULT_NUMBER_OF_ACCOUNTS: usize = 5;

impl GenesisConfig {
    pub fn for_local_testing() -> Self {
        Self::custom_genesis(DEFAULT_NUMBER_OF_ACCOUNTS)
    }

    pub fn custom_genesis(num_accounts: usize) -> Self {
        let mut accounts = Vec::new();
        for _ in 0..num_accounts {
            accounts.push(AccountConfig { address: None })
        }

        Self {
            accounts,
            ..Default::default()
        }
    }

    pub fn generate_accounts<R: rand::RngCore + rand::CryptoRng>(
        &self,
        mut rng: R,
    ) -> Result<Vec<SomaKeyPair>> {
        let mut addresses = Vec::new();

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
        }

        Ok(keys)
    }
}

#[derive(Default)]
pub struct ValidatorGenesisConfigBuilder {
    protocol_key_pair: Option<AuthorityKeyPair>,
    account_key_pair: Option<SomaKeyPair>,
    ip: Option<String>,
    /// If set, the validator will use deterministic addresses based on the port offset.
    /// This is useful for benchmarking.
    port_offset: Option<u16>,
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

    pub fn with_deterministic_ports(mut self, port_offset: u16) -> Self {
        self.port_offset = Some(port_offset);
        self
    }

    pub fn build<R: rand::RngCore + rand::CryptoRng>(self, rng: &mut R) -> ValidatorGenesisConfig {
        let ip = self.ip.unwrap_or_else(local_ip_utils::get_new_ip);
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

        let (network_address, consensus_address) = if let Some(offset) = self.port_offset {
            (
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset),
                // local_ip_utils::new_deterministic_udp_address_for_testing(&ip, offset + 1),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 2),
            )
        } else {
            (
                local_ip_utils::new_tcp_address_for_testing(&ip),
                // local_ip_utils::new_udp_address_for_testing(&ip),
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
        }
    }
}
