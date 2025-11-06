use crate::base::SomaAddress;
use crate::config::{PersistedConfig, SOMA_KEYSTORE_FILENAME};
use crate::crypto::{NetworkKeyPair, NetworkPublicKey};
use crate::parameters::{HttpParameters, TonicParameters};
use crate::shard_crypto::keys::{EncoderKeyPair, EncoderPublicKey};
use crate::{
    crypto::{get_key_pair_from_rng, EncodeDecodeBase64, SomaKeyPair},
    genesis::Genesis,
    multiaddr::Multiaddr,
};
use anyhow::anyhow;
use fastcrypto::{bls12381::min_sig::BLS12381KeyPair, traits::KeyPair};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::path::Path;
use std::{
    net::SocketAddr,
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use super::local_ip_utils;

/// Represents configuration options for creating encoder committees
pub enum EncoderCommitteeConfig {
    Size(NonZeroUsize),
    Encoders(Vec<EncoderGenesisConfig>),
    EncoderKeys(Vec<EncoderKeyPair>),
    Deterministic((NonZeroUsize, Option<Vec<EncoderKeyPair>>)),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub account_keypair: KeyPairWithPath,
    /// Keys for the encoder protocol
    pub encoder_keypair: EncoderKeyPairWithPath,
    /// Keys for network peer identification
    pub network_keypair: KeyPairWithPath,
    pub internal_network_address: Multiaddr,
    pub external_network_address: Multiaddr,
    /// The network address for the unencrypted object storage
    pub internal_object_address: Multiaddr,
    /// The network address for object storage
    pub external_object_address: Multiaddr,
    /// The network address for inference service
    pub inference_address: Multiaddr,
    /// The network address for evaluation service
    pub evaluation_address: Multiaddr,
    /// Parameters for the encoder system
    // TODO: pub parameters: Arc<Parameters>,
    /// Parameters for the object system
    pub object_parameters: Arc<HttpParameters>,
    /// Parameters for the evaluation system
    pub evaluation_parameters: Arc<TonicParameters>,
    /// Path to the project root for Python interpreter
    pub project_root: PathBuf,
    /// Path to the entry point for Python module
    pub entry_point: PathBuf,

    /// Address of the validator node for fetching committees
    pub validator_sync_address: Multiaddr,

    pub rpc_address: SocketAddr,

    /// Genesis for blockchain, including validator and encoder committees
    pub genesis: Genesis,

    pub epoch_duration_ms: u64,

    pub db_path: PathBuf,
}

impl EncoderConfig {
    /// Creates a new EncoderConfig with the provided keypairs and addresses
    pub fn new(
        soma_keypair: SomaKeyPair,
        encoder_keypair: EncoderKeyPair,
        network_keypair: NetworkKeyPair,
        internal_network_address: Multiaddr,
        external_network_address: Multiaddr,
        internal_object_address: Multiaddr,
        external_object_address: Multiaddr,
        inference_address: Multiaddr,
        evaluation_address: Multiaddr,
        rpc_address: SocketAddr,
        project_root: PathBuf,
        entry_point: PathBuf,
        validator_sync_address: Multiaddr,
        genesis: Genesis,
        db_path: PathBuf,
    ) -> Self {
        // Create default parameters
        // TODO: let parameters = Arc::new(Parameters::default());
        let object_parameters = Arc::new(HttpParameters::default());
        let evaluation_parameters = Arc::new(TonicParameters::default());

        Self {
            account_keypair: KeyPairWithPath::new(soma_keypair),
            encoder_keypair: EncoderKeyPairWithPath::new(encoder_keypair),
            network_keypair: KeyPairWithPath::new(SomaKeyPair::Ed25519(
                network_keypair.into_inner().copy(),
            )),
            internal_network_address,
            external_network_address,
            internal_object_address,
            external_object_address,
            inference_address,
            evaluation_address,
            // parameters,
            db_path,
            object_parameters,
            evaluation_parameters,
            project_root,
            entry_point,
            validator_sync_address,
            rpc_address,
            genesis,
            epoch_duration_ms: 1000, //TODO: Default epoch duration
        }
    }

    pub fn db_path(&self) -> PathBuf {
        self.db_path.clone()
    }

    pub fn protocol_public_key(&self) -> EncoderPublicKey {
        self.protocol_key_pair().public().into()
    }

    pub fn protocol_key_pair(&self) -> &EncoderKeyPair {
        self.encoder_keypair.encoder_keypair()
    }

    pub fn network_public_key(&self) -> NetworkPublicKey {
        NetworkPublicKey::new(
            self.network_keypair
                .keypair()
                .inner()
                .copy()
                .public()
                .clone(),
        )
    }

    /// Sets the epoch duration in milliseconds
    pub fn with_epoch_duration(mut self, duration_ms: u64) -> Self {
        self.epoch_duration_ms = duration_ms;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyPairWithPath {
    #[serde(flatten)]
    location: KeyPairLocation,
    #[serde(skip)]
    keypair: OnceLock<Arc<SomaKeyPair>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde_as]
#[serde(untagged)]
enum KeyPairLocation {
    InPlace {
        #[serde_as(as = "Arc<KeyPairBase64>")]
        value: Arc<SomaKeyPair>,
    },
    File {
        #[serde(rename = "path")]
        path: PathBuf,
    },
}
// }

impl KeyPairWithPath {
    pub fn new(kp: SomaKeyPair) -> Self {
        let cell: OnceLock<Arc<SomaKeyPair>> = OnceLock::new();
        let arc_kp = Arc::new(kp);
        // OK to unwrap panic because authority should not start without all keypairs loaded.
        cell.set(arc_kp.clone()).expect("Failed to set keypair");
        Self {
            location: KeyPairLocation::InPlace { value: arc_kp },
            keypair: cell,
        }
    }

    //     pub fn new_from_path(path: PathBuf) -> Self {
    //         let cell: OnceCell<Arc<SomaKeyPair>> = OnceCell::new();
    //         // OK to unwrap panic because authority should not start without all keypairs loaded.
    //         cell.set(Arc::new(read_keypair_from_file(&path).unwrap_or_else(
    //             |e| panic!("Invalid keypair file at path {:?}: {e}", &path),
    //         )))
    //         .expect("Failed to set keypair");
    //         Self {
    //             location: KeyPairLocation::File { path },
    //             keypair: cell,
    //         }
    //     }

    pub fn keypair(&self) -> &SomaKeyPair {
        self.keypair
            .get_or_init(|| match &self.location {
                KeyPairLocation::InPlace { value } => value.clone(),
                KeyPairLocation::File { path } => {
                    // OK to unwrap panic because authority should not start without all keypairs loaded.
                    Arc::new(
                        read_keypair_from_file(path).unwrap_or_else(|e| {
                            panic!("Invalid keypair file at path {:?}: {e}", path)
                        }),
                    )
                }
            })
            .as_ref()
    }
}

/// Wrapper struct for AuthorityKeyPair that can be deserialized from a file path.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct EncoderKeyPairWithPath {
    // #[serde(flatten)]
    // location: AuthorityKeyPairLocation,
    #[serde(skip)]
    keypair: OnceLock<Arc<EncoderKeyPair>>,
}

// #[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Eq)]
// #[serde_as]
// #[serde(untagged)]
// enum EncoderKeyPairLocation {
//     InPlace { value: Arc<EncoderKeyPair> },
//     File { path: PathBuf },
// }

impl EncoderKeyPairWithPath {
    pub fn new(kp: EncoderKeyPair) -> Self {
        let cell: OnceLock<Arc<EncoderKeyPair>> = OnceLock::new();
        let arc_kp = Arc::new(kp);
        // OK to unwrap panic because authority should not start without all keypairs loaded.
        cell.set(arc_kp.clone())
            .expect("Failed to set authority keypair");
        Self { keypair: cell }
    }

    pub fn encoder_keypair(&self) -> &EncoderKeyPair {
        self.keypair.get().unwrap().as_ref()
    }
}

/// Read from file as Base64 encoded `privkey` and return a AuthorityKeyPair.
pub fn read_encoder_keypair_from_file<P: AsRef<std::path::Path>>(
    path: P,
) -> anyhow::Result<EncoderKeyPair> {
    let contents = std::fs::read_to_string(path)?;
    let kp = BLS12381KeyPair::decode_base64(contents.as_str().trim()).map_err(|e| anyhow!(e))?;
    Ok(EncoderKeyPair::new(kp))
}

/// Read from file as Base64 encoded `flag || privkey` and return a SomaKeypair.
pub fn read_keypair_from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<SomaKeyPair> {
    let contents = std::fs::read_to_string(path)?;
    SomaKeyPair::decode_base64(contents.as_str().trim()).map_err(|e| anyhow!(e))
}

// 1. Create EncoderGenesisConfig struct similar to ValidatorGenesisConfig
#[derive(Serialize, Deserialize)]
pub struct EncoderGenesisConfig {
    pub encoder_key_pair: EncoderKeyPair,
    pub account_key_pair: SomaKeyPair,
    pub network_key_pair: NetworkKeyPair,
    pub internal_network_address: Multiaddr,
    pub external_network_address: Multiaddr,
    pub object_address: Multiaddr,
    pub local_object_address: Multiaddr,
    pub inference_address: Multiaddr,
    pub evaluation_address: Multiaddr,
    pub stake: u64,
    pub commission_rate: u64,
    pub byte_price: u64,
}

// 2. Create a builder for EncoderGenesisConfig
#[derive(Default)]
pub struct EncoderGenesisConfigBuilder {
    encoder_key_pair: Option<EncoderKeyPair>,
    account_key_pair: Option<SomaKeyPair>,
    network_key_pair: Option<NetworkKeyPair>,
    ip: Option<String>,
    port_offset: Option<u16>,
    stake: Option<u64>,
}

impl EncoderGenesisConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_encoder_key_pair(mut self, key_pair: EncoderKeyPair) -> Self {
        self.encoder_key_pair = Some(key_pair);
        self
    }

    pub fn with_account_key_pair(mut self, key_pair: SomaKeyPair) -> Self {
        self.account_key_pair = Some(key_pair);
        self
    }

    pub fn with_network_key_pair(mut self, key_pair: NetworkKeyPair) -> Self {
        self.network_key_pair = Some(key_pair);
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

    pub fn with_stake(mut self, stake: u64) -> Self {
        self.stake = Some(stake);
        self
    }

    pub fn build<R: rand::RngCore + rand::CryptoRng>(self, rng: &mut R) -> EncoderGenesisConfig {
        let ip = self.ip.unwrap_or_else(local_ip_utils::get_new_ip);
        let stake = self.stake.unwrap_or(default_encoder_stake());

        // Generate or use provided keypairs
        let encoder_key_pair = self
            .encoder_key_pair
            .unwrap_or_else(|| EncoderKeyPair::new(get_key_pair_from_rng(rng).1));

        let account_key_pair = self
            .account_key_pair
            .unwrap_or_else(|| SomaKeyPair::Ed25519(get_key_pair_from_rng(rng).1));

        let network_key_pair = self
            .network_key_pair
            .unwrap_or_else(|| NetworkKeyPair::new(get_key_pair_from_rng(rng).1));

        // Generate network addresses
        let (
            internal_network_address,
            external_network_address,
            object_address,
            local_object_address,
            inference_address,
            evaluation_address,
        ) = if let Some(offset) = self.port_offset {
            (
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 1),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 2),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 3),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 4),
                local_ip_utils::new_deterministic_tcp_address_for_testing(&ip, offset + 5),
            )
        } else {
            (
                local_ip_utils::new_tcp_address_for_testing(&ip),
                local_ip_utils::new_tcp_address_for_testing(&ip),
                local_ip_utils::new_tcp_address_for_testing(&ip),
                local_ip_utils::new_tcp_address_for_testing(&ip),
                local_ip_utils::new_tcp_address_for_testing(&ip),
                local_ip_utils::new_tcp_address_for_testing(&ip),
            )
        };

        EncoderGenesisConfig {
            encoder_key_pair,
            account_key_pair,
            network_key_pair,
            internal_network_address,
            external_network_address,
            object_address,
            local_object_address,
            inference_address,
            evaluation_address,
            stake,
            commission_rate: DEFAULT_ENCODER_COMMISSION_RATE,
            byte_price: DEFAULT_ENCODER_BYTE_PRICE,
        }
    }
}

// 3. Add default constants for encoders
const DEFAULT_ENCODER_COMMISSION_RATE: u64 = 200;
const DEFAULT_ENCODER_BYTE_PRICE: u64 = 1000;

fn default_encoder_stake() -> u64 {
    20_000_000_000_000_000 // Same as validator default stake
}
