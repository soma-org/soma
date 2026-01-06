use crate::base::SomaAddress;
use crate::crypto::NetworkPublicKey;
use crate::metadata::DownloadMetadata;
use crate::multiaddr::Multiaddr;
use crate::shard_crypto::keys::EncoderPublicKey;
use serde::{Deserialize, Serialize};

/// Public information about an encoder for genesis ceremony.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncoderInfo {
    pub name: String,
    pub account_address: SomaAddress,
    pub encoder_pubkey: EncoderPublicKey,
    pub network_key: NetworkPublicKey,
    pub internal_network_address: Multiaddr,
    pub external_network_address: Multiaddr,
    pub object_address: Multiaddr,
    pub commission_rate: u64,
    pub byte_price: u64,
}

/// Encoder info bundled with probe metadata for genesis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenesisEncoderInfo {
    pub info: EncoderInfo,
    pub probe: DownloadMetadata,
}

impl GenesisEncoderInfo {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.info.name.is_empty() {
            anyhow::bail!("Encoder name cannot be empty");
        }
        Ok(())
    }
}

// Conversion from EncoderGenesisConfig (for local testing)
impl From<&crate::config::encoder_config::EncoderGenesisConfig> for EncoderInfo {
    fn from(config: &crate::config::encoder_config::EncoderGenesisConfig) -> Self {
        use crate::crypto::PublicKey;

        let account_address = SomaAddress::from(&config.account_key_pair.public());

        Self {
            name: account_address.to_string(),
            account_address,
            encoder_pubkey: config.encoder_key_pair.public().clone(),
            network_key: config.network_key_pair.public().clone(),
            internal_network_address: config.internal_network_address.clone(),
            external_network_address: config.external_network_address.clone(),
            object_address: config.object_address.clone(),
            commission_rate: config.commission_rate,
            byte_price: config.byte_price,
        }
    }
}

impl From<&crate::config::encoder_config::EncoderGenesisConfig> for GenesisEncoderInfo {
    fn from(config: &crate::config::encoder_config::EncoderGenesisConfig) -> Self {
        Self {
            info: EncoderInfo::from(config),
            probe: config.probe.clone(),
        }
    }
}
