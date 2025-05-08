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
    /// Create a deterministic committee with optional provided keys
    Deterministic((NonZeroUsize, Option<Vec<EncoderKeyPair>>)),
}

/// Configuration for a single encoder node in a simulation environment
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
    /// Information about the network topology
    pub networking_info: NetworkingInfo,
    /// Mappings between peer keys and encoder keys
    pub connections_info: ConnectionsInfo,
    /// Context for the node
    pub context: Context,
    /// Set of allowed public keys for TLS verification
    pub allowed_public_keys: AllowPublicKeys,

    /// Address of the validator node for fetching committees
    pub validator_rpc_address: Option<Multiaddr>,

    /// Genesis committee for committee verification
    pub genesis_committee: Option<Committee>,
}

impl EncoderConfig {
    /// Creates a new EncoderConfig with the provided keypairs and addresses
    pub fn new(
        encoder_keypair: EncoderKeyPair,
        peer_keypair: PeerKeyPair,
        ip: IpAddr,
        internal_network_address: Multiaddr,
        external_network_address: Multiaddr,
        object_address: Multiaddr,
        probe_address: Multiaddr,
        project_root: PathBuf,
        entry_point: PathBuf,
    ) -> Self {
        // Create default parameters
        let parameters = Arc::new(Parameters::default());
        let object_parameters = Arc::new(objects::parameters::Parameters::default());
        let probe_parameters = Arc::new(probe::parameters::Parameters::default());

        // Create empty network info
        let networking_info = NetworkingInfo::default();
        let connections_info = ConnectionsInfo::new(BTreeMap::new());

        // Create a minimal context for testing with only this encoder
        let context = Self::create_test_context(
            &encoder_keypair,
            vec![encoder_keypair.public()],
            0,
            HashMap::new(),
        );

        // Create initial AllowPublicKeys with just this node's key
        let mut allowed_keys = BTreeSet::new();
        allowed_keys.insert(peer_keypair.public().into_inner());
        let allowed_public_keys = AllowPublicKeys::new(allowed_keys);

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
            networking_info,
            connections_info,
            context,
            allowed_public_keys,
            genesis_committee: None,
            validator_rpc_address: None,
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

    /// Sets the validator node address
    pub fn with_validator_rpc_address(mut self, address: Multiaddr) -> Self {
        self.validator_rpc_address = Some(address);
        self
    }

    /// Sets the genesis committee
    pub fn with_genesis_committee(mut self, committee: Committee) -> Self {
        self.genesis_committee = Some(committee);
        self
    }

    /// Creates a test context for a group of encoders
    pub fn create_test_context(
        own_encoder_keypair: &EncoderKeyPair,
        all_encoder_keys: Vec<EncoderPublicKey>,
        own_index: usize,
        encoder_object_servers: HashMap<
            EncoderPublicKey,
            (PeerPublicKey, soma_network::multiaddr::Multiaddr),
        >,
    ) -> Context {
        // Create encoder committee with all encoders
        let encoder_committee = EncoderCommittee::new_for_testing(all_encoder_keys);
        let (authority_committee, _) =
            AuthorityCommittee::local_test_committee(0, vec![1, 1, 1, 1]);

        let encoder_index = EncoderIndex::new(own_index as u32);
        let committees = Committees::new(
            0, // epoch
            authority_committee,
            encoder_committee,
            encoder_index,
            1, // vdf_iterations
        );

        // Create inner context with the provided object servers
        let inner_context = InnerContext::new(
            [committees.clone(), committees],
            0,
            own_encoder_keypair.public(),
            encoder_object_servers,
        );

        Context::new(inner_context)
    }
}
