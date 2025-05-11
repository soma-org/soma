use anyhow::anyhow;
use encoder::{
    messaging::tonic::{internal::ConnectionsInfo, NetworkingInfo},
    types::{
        context::{Committees, Context, InnerContext},
        encoder_committee::{EncoderCommittee, EncoderIndex},
        parameters::Parameters,
    },
};
use fastcrypto::{bls12381::min_sig::BLS12381KeyPair, traits::KeyPair};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use shared::{
    authority_committee::AuthorityCommittee,
    crypto::keys::{EncoderKeyPair, EncoderPublicKey, PeerKeyPair, PeerPublicKey},
};
use soma_tls::AllowPublicKeys;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    net::{IpAddr, Ipv4Addr},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};
use types::{
    committee::Committee,
    crypto::{EncodeDecodeBase64, SomaKeyPair},
    multiaddr::Multiaddr,
};

/// Represents configuration options for creating encoder committees
pub enum EncoderCommitteeConfig {
    /// Create a committee with a specific size (number of encoders)
    Size(NonZeroUsize),
    /// Create a committee with specific encoder configurations
    Encoders(Vec<EncoderConfig>),
    /// Create a committee using existing encoder keypairs
    EncoderKeys(Vec<EncoderKeyPair>),
    // Create a deterministic committee with optional provided keys
    // Deterministic((NonZeroUsize, Option<Vec<EncoderKeyPair>>)),
}

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub account_keypair: KeyPairWithPath,
    /// Keys for the encoder protocol
    pub encoder_keypair: EncoderKeyPairWithPath,
    /// Keys for network peer identification
    pub peer_keypair: KeyPairWithPath,
    pub internal_network_address: Multiaddr,
    pub external_network_address: Multiaddr,
    /// The network address for object storage
    pub object_address: Multiaddr,
    /// The network address for probe service
    pub probe_address: Multiaddr,
    /// Parameters for the encoder system
    pub parameters: Arc<Parameters>,
    /// Parameters for the object system
    pub object_parameters: Arc<objects::parameters::Parameters>,
    /// Parameters for the probe system
    pub probe_parameters: Arc<probe::parameters::Parameters>,
    /// Path to the project root for Python interpreter
    pub project_root: PathBuf,
    /// Path to the entry point for Python module
    pub entry_point: PathBuf,

    /// Address of the validator node for fetching committees
    pub validator_rpc_address: Multiaddr,

    /// Genesis committee for committee verification
    pub genesis_committee: Committee,

    pub epoch_duration_ms: u64,
}

impl EncoderConfig {
    /// Creates a new EncoderConfig with the provided keypairs and addresses
    pub fn new(
        soma_keypair: SomaKeyPair,
        encoder_keypair: EncoderKeyPair,
        peer_keypair: PeerKeyPair,
        internal_network_address: Multiaddr,
        external_network_address: Multiaddr,
        object_address: Multiaddr,
        probe_address: Multiaddr,
        project_root: PathBuf,
        entry_point: PathBuf,
        validator_rpc_address: Multiaddr,
        genesis_committee: Committee,
    ) -> Self {
        // Create default parameters
        let parameters = Arc::new(Parameters::default());
        let object_parameters = Arc::new(objects::parameters::Parameters::default());
        let probe_parameters = Arc::new(probe::parameters::Parameters::default());

        Self {
            account_keypair: KeyPairWithPath::new(soma_keypair),
            encoder_keypair: EncoderKeyPairWithPath::new(encoder_keypair),
            peer_keypair: KeyPairWithPath::new(SomaKeyPair::Ed25519(peer_keypair.inner().copy())),
            internal_network_address,
            external_network_address,
            object_address,
            probe_address,
            parameters,
            object_parameters,
            probe_parameters,
            project_root,
            entry_point,
            validator_rpc_address,
            genesis_committee,
            epoch_duration_ms: 1000, //TODO: Default epoch duration
        }
    }

    pub fn protocol_public_key(&self) -> EncoderPublicKey {
        self.protocol_key_pair().public().into()
    }

    pub fn protocol_key_pair(&self) -> &EncoderKeyPair {
        self.encoder_keypair.encoder_keypair()
    }

    pub fn peer_public_key(&self) -> PeerPublicKey {
        PeerPublicKey::new(self.peer_keypair.keypair().inner().copy().public().clone())
    }

    /// Sets the epoch duration in milliseconds
    pub fn with_epoch_duration(mut self, duration_ms: u64) -> Self {
        self.epoch_duration_ms = duration_ms;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyPairWithPath {
    // #[serde(flatten)]
    location: KeyPairLocation,
    #[serde(skip)]
    keypair: OnceLock<Arc<SomaKeyPair>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde_as]
enum KeyPairLocation {
    #[serde(skip)]
    InPlace {
        // #[serde_as(as = "Arc<KeyPairBase64>")]
        value: Arc<SomaKeyPair>,
    },
    File {
        // #[serde(rename = "path")]
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

/// Read from file as Base64 encoded `flag || privkey` and return a SuiKeypair.
pub fn read_keypair_from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<SomaKeyPair> {
    let contents = std::fs::read_to_string(path)?;
    SomaKeyPair::decode_base64(contents.as_str().trim()).map_err(|e| anyhow!(e))
}
