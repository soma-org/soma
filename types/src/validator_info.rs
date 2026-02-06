use crate::base::SomaAddress;
use crate::crypto::{AuthorityPublicKey, AuthorityPublicKeyBytes, NetworkPublicKey};
use crate::multiaddr::Multiaddr;
use anyhow::bail;
use fastcrypto::traits::ToFromBytes;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

const MAX_VALIDATOR_METADATA_LENGTH: usize = 256;

/// Publicly known information about a validator
#[serde_as]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub struct ValidatorInfo {
    pub account_address: SomaAddress,
    pub protocol_key: AuthorityPublicKeyBytes,
    pub worker_key: NetworkPublicKey,
    pub network_key: NetworkPublicKey,
    pub commission_rate: u64,
    pub network_address: Multiaddr,
    pub p2p_address: Multiaddr,
    pub primary_address: Multiaddr,
}

impl ValidatorInfo {
    pub fn soma_address(&self) -> SomaAddress {
        self.account_address
    }

    pub fn protocol_key(&self) -> AuthorityPublicKeyBytes {
        self.protocol_key
    }

    pub fn worker_key(&self) -> &NetworkPublicKey {
        &self.worker_key
    }

    pub fn network_key(&self) -> &NetworkPublicKey {
        &self.network_key
    }

    pub fn commission_rate(&self) -> u64 {
        self.commission_rate
    }

    pub fn network_address(&self) -> &Multiaddr {
        &self.network_address
    }

    pub fn primary_address(&self) -> &Multiaddr {
        &self.primary_address
    }

    pub fn p2p_address(&self) -> &Multiaddr {
        &self.p2p_address
    }
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenesisValidatorInfo {
    pub info: ValidatorInfo,
}

impl GenesisValidatorInfo {
    pub fn validate(&self) -> anyhow::Result<(), anyhow::Error> {
        if !self.info.network_address.to_string().is_ascii() {
            bail!("network address must be ascii");
        }
        if self.info.network_address.len() > MAX_VALIDATOR_METADATA_LENGTH {
            bail!("network address must be <= {MAX_VALIDATOR_METADATA_LENGTH} bytes long");
        }

        if !self.info.p2p_address.to_string().is_ascii() {
            bail!("p2p address must be ascii");
        }
        if self.info.p2p_address.len() > MAX_VALIDATOR_METADATA_LENGTH {
            bail!("p2p address must be <= {MAX_VALIDATOR_METADATA_LENGTH} bytes long");
        }

        if !self.info.primary_address.to_string().is_ascii() {
            bail!("primary address must be ascii");
        }
        if self.info.primary_address.len() > MAX_VALIDATOR_METADATA_LENGTH {
            bail!("primary address must be <= {MAX_VALIDATOR_METADATA_LENGTH} bytes long");
        }

        if self.info.commission_rate > 10000 {
            bail!("commissions rate must be lower than 100%");
        }

        Ok(())
    }
}

// Conversion from ValidatorGenesisConfig (for local testing)
impl From<&crate::config::genesis_config::ValidatorGenesisConfig> for ValidatorInfo {
    fn from(config: &crate::config::genesis_config::ValidatorGenesisConfig) -> Self {
        use fastcrypto::traits::KeyPair;

        // account_key_pair is SomaKeyPair, .public() returns PublicKey enum
        let account_address = SomaAddress::from(&config.account_key_pair.public());

        // key_pair is AuthorityKeyPair (BLS12381KeyPair)
        // .public() returns &AuthorityPublicKey (BLS12381PublicKey)
        // Convert to AuthorityPublicKeyBytes
        let protocol_key: AuthorityPublicKeyBytes = config.key_pair.public().into();

        // worker_key_pair is NetworkKeyPair, .public() returns NetworkPublicKey
        let worker_key = config.worker_key_pair.public();

        // network_key_pair is NetworkKeyPair, .public() returns NetworkPublicKey
        let network_key = config.network_key_pair.public();

        Self {
            account_address,
            protocol_key,
            worker_key,
            network_key,
            network_address: config.network_address.clone(),
            p2p_address: config.p2p_address.clone(),
            primary_address: config.consensus_address.clone(), // consensus_address maps to primary_address
            commission_rate: config.commission_rate,
        }
    }
}

impl From<&crate::config::genesis_config::ValidatorGenesisConfig> for GenesisValidatorInfo {
    fn from(config: &crate::config::genesis_config::ValidatorGenesisConfig) -> Self {
        Self { info: ValidatorInfo::from(config) }
    }
}

impl From<GenesisValidatorInfo> for GenesisValidatorMetadata {
    fn from(GenesisValidatorInfo { info }: GenesisValidatorInfo) -> Self {
        Self {
            soma_address: info.account_address,
            commission_rate: info.commission_rate,
            protocol_public_key: info.protocol_key.as_ref().to_vec(),
            network_public_key: info.network_key.to_bytes().to_vec(),
            worker_public_key: info.worker_key.to_bytes().to_vec(),
            network_address: info.network_address,
            p2p_address: info.p2p_address,
            primary_address: info.primary_address,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct GenesisValidatorMetadata {
    pub soma_address: SomaAddress,
    pub commission_rate: u64,
    pub protocol_public_key: Vec<u8>, // BLS12381 public key bytes
    pub network_public_key: Vec<u8>,  // Ed25519 public key bytes
    pub worker_public_key: Vec<u8>,   // Ed25519 public key bytes
    pub network_address: Multiaddr,
    pub p2p_address: Multiaddr,
    pub primary_address: Multiaddr,
}
