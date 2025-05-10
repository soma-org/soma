use encoder::{
    messaging::tonic::{internal::ConnectionsInfo, NetworkingInfo},
    types::{
        context::{Committees, Context, InnerContext},
        encoder_committee::{EncoderCommittee, EncoderIndex},
        parameters::Parameters,
    },
};
use rand::{CryptoRng, RngCore};
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
    sync::Arc,
};
use types::{committee::Committee, multiaddr::Multiaddr};

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

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    /// Keys for the encoder protocol
    pub encoder_keypair: EncoderKeyPair,
    /// Keys for network peer identification
    pub peer_keypair: PeerKeyPair,
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
            encoder_keypair,
            peer_keypair,
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
            epoch_duration_ms: 1000, // Default epoch duration
        }
    }

    /// Returns the encoder public key
    pub fn protocol_public_key(&self) -> EncoderPublicKey {
        self.encoder_keypair.public()
    }

    /// Returns the peer public key
    pub fn peer_public_key(&self) -> PeerPublicKey {
        self.peer_keypair.public()
    }

    /// Returns the object server information for this encoder
    pub fn get_object_server_info(&self) -> (EncoderPublicKey, Multiaddr) {
        (self.protocol_public_key(), self.object_address.clone())
    }

    /// Sets the epoch duration in milliseconds
    pub fn with_epoch_duration(mut self, duration_ms: u64) -> Self {
        self.epoch_duration_ms = duration_ms;
        self
    }
}
