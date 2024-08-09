use crate::{
    network::multiaddr::Multiaddr, NetworkKeyPair, NetworkPublicKey, ProtocolKeyPair,
    ProtocolPublicKey,
};
use std::{
    fmt::{Display, Formatter},
    net::{TcpListener, TcpStream},
    ops::{Index, IndexMut},
};

use rand::{rngs::StdRng, SeedableRng as _};
use serde::{Deserialize, Serialize};

pub type Epoch = u64;

/// Voting power of an authority, roughly proportional to the actual amount staked
/// by the authority.
/// Total stake / voting power of all authorities should sum to 10,000.
pub type Stake = u64;

#[derive(Clone, Debug)]
pub struct Committee {
    /// The epoch number of this committee
    epoch: Epoch,

    /// Total stake in the committee.
    total_stake: Stake,

    /// Protocol and network info of each authority.
    authorities: Vec<Authority>,

    /// The quorum threshold (2f+1).
    quorum_threshold: Stake,
    /// The validity threshold (f+1).
    validity_threshold: Stake,
}

impl Committee {
    pub fn new(epoch: Epoch, authorities: Vec<Authority>) -> Self {
        assert!(!authorities.is_empty(), "Committee cannot be empty!");
        assert!(
            authorities.len() < u32::MAX as usize,
            "Too many authorities ({})!",
            authorities.len()
        );
        let total_stake = authorities.iter().map(|a| a.stake).sum();
        assert_ne!(total_stake, 0, "Total stake cannot be zero!");
        let quorum_threshold = 2 * total_stake / 3 + 1;
        let validity_threshold = (total_stake + 2) / 3;
        Self {
            epoch,
            authorities,
            total_stake,
            quorum_threshold,
            validity_threshold,
        }
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    pub fn total_stake(&self) -> Stake {
        self.total_stake
    }

    pub fn quorum_threshold(&self) -> Stake {
        self.quorum_threshold
    }

    pub fn validity_threshold(&self) -> Stake {
        self.validity_threshold
    }

    pub fn stake(&self, authority_index: AuthorityIndex) -> Stake {
        self.authorities[authority_index].stake
    }

    pub fn authority(&self, authority_index: AuthorityIndex) -> &Authority {
        &self.authorities[authority_index]
    }

    pub fn authorities(&self) -> impl Iterator<Item = (AuthorityIndex, &Authority)> {
        self.authorities
            .iter()
            .enumerate()
            .map(|(i, a)| (AuthorityIndex(i as u32), a))
    }

    /// Returns true if the provided stake has reached quorum (2f+1).
    pub fn reached_quorum(&self, stake: Stake) -> bool {
        stake >= self.quorum_threshold()
    }

    /// Returns true if the provided stake has reached validity (f+1).
    pub fn reached_validity(&self, stake: Stake) -> bool {
        stake >= self.validity_threshold()
    }

    /// Coverts an index to an AuthorityIndex, if valid.
    /// Returns None if index is out of bound.
    pub fn to_authority_index(&self, index: usize) -> Option<AuthorityIndex> {
        if index < self.authorities.len() {
            Some(AuthorityIndex(index as u32))
        } else {
            None
        }
    }

    /// Returns true if the provided index is valid.
    pub fn is_valid_index(&self, index: AuthorityIndex) -> bool {
        index.value() < self.size()
    }

    /// Returns number of authorities in the committee.
    pub fn size(&self) -> usize {
        self.authorities.len()
    }
}

#[derive(Debug, Clone)]
pub struct Authority {
    /// Voting power of the authority in the committee.
    pub stake: Stake,

    /// Network address for communicating with the authority.
    pub address: Multiaddr,

    /// The authority's hostname, for metrics and logging.
    pub hostname: String,

    /// The authority's public key for verifying blocks.
    pub protocol_key: ProtocolPublicKey,

    /// The authority's public key for TLS and as network identity.
    pub network_key: NetworkPublicKey,
}

#[derive(
    Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug, Default, Hash, Serialize, Deserialize,
)]
pub struct AuthorityIndex(u32);

impl AuthorityIndex {
    pub const ZERO: Self = Self(0);
    pub const MIN: Self = Self::ZERO;
    pub const MAX: Self = Self(u32::MAX);

    pub fn value(&self) -> usize {
        self.0 as usize
    }

    pub fn new_for_test(index: u32) -> Self {
        Self(index)
    }
}

impl Display for AuthorityIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.value() < 26 {
            let c = (b'A' + self.value() as u8) as char;
            f.write_str(c.to_string().as_str())
        } else {
            write!(f, "[{:02}]", self.value())
        }
    }
}

impl<T> Index<AuthorityIndex> for Vec<T> {
    type Output = T;

    fn index(&self, index: AuthorityIndex) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T> IndexMut<AuthorityIndex> for Vec<T> {
    fn index_mut(&mut self, index: AuthorityIndex) -> &mut Self::Output {
        self.get_mut(index.value()).unwrap()
    }
}

/// Creates a committee for local testing, and the corresponding key pairs for the authorities.
pub fn local_committee_and_keys(
    epoch: Epoch,
    authorities_stake: Vec<Stake>,
) -> (Committee, Vec<(NetworkKeyPair, ProtocolKeyPair)>) {
    let mut authorities = vec![];
    let mut key_pairs = vec![];
    let mut rng = StdRng::from_seed([0; 32]);
    for (i, stake) in authorities_stake.into_iter().enumerate() {
        // let authority_keypair = AuthorityKeyPair::generate(&mut rng);
        let protocol_keypair = ProtocolKeyPair::generate(&mut rng);
        let network_keypair = NetworkKeyPair::generate(&mut rng);
        authorities.push(Authority {
            stake,
            address: get_available_local_address(),
            hostname: format!("test_host_{i}").to_string(),
            // authority_key: authority_keypair.public(),
            protocol_key: protocol_keypair.public(),
            network_key: network_keypair.public(),
        });
        key_pairs.push((network_keypair, protocol_keypair));
    }

    let committee = Committee::new(epoch, authorities);
    (committee, key_pairs)
}

/// Returns a local address with an ephemeral port.
fn get_available_local_address() -> Multiaddr {
    let host = "127.0.0.1";
    let port = get_available_port(host);
    format!("/ip4/{}/udp/{}", host, port).parse().unwrap()
}

/// Returns an ephemeral, available port. On unix systems, the port returned will be in the
/// TIME_WAIT state ensuring that the OS won't hand out this port for some grace period.
/// Callers should be able to bind to this port given they use SO_REUSEADDR.
fn get_available_port(host: &str) -> u16 {
    const MAX_PORT_RETRIES: u32 = 1000;

    for _ in 0..MAX_PORT_RETRIES {
        if let Ok(port) = get_ephemeral_port(host) {
            return port;
        }
    }

    panic!("Error: could not find an available port");
}

fn get_ephemeral_port(host: &str) -> std::io::Result<u16> {
    // Request a random available port from the OS
    let listener = TcpListener::bind((host, 0))?;
    let addr = listener.local_addr()?;

    // Create and accept a connection (which we'll promptly drop) in order to force the port
    // into the TIME_WAIT state, ensuring that the port will be reserved from some limited
    // amount of time (roughly 60s on some Linux systems)
    let _sender = TcpStream::connect(addr)?;
    let _incoming = listener.accept()?;

    Ok(addr.port())
}
