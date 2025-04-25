use crate::base::AuthorityName;
use crate::{committee::Authority, crypto::AuthorityKeyPair, multiaddr::Multiaddr};
use crate::{
    committee::{Committee, Epoch, Stake},
    crypto::{NetworkKeyPair, NetworkPublicKey, ProtocolKeyPair, ProtocolPublicKey},
};
use fastcrypto::traits::KeyPair;
use std::collections::BTreeMap;
use std::{
    fmt::{Display, Formatter},
    net::{TcpListener, TcpStream},
    ops::{Index, IndexMut},
};

use rand::{rngs::StdRng, SeedableRng as _};
use serde::{Deserialize, Serialize};

/// Creates a committee for local testing, and the corresponding key pairs for the authorities.
pub fn local_committee_and_keys(
    epoch: Epoch,
    authorities_stake: Vec<Stake>,
) -> (
    Committee,
    Vec<(NetworkKeyPair, ProtocolKeyPair)>,
    Vec<AuthorityKeyPair>,
) {
    let mut authorities = BTreeMap::new();
    let mut voting_weights = BTreeMap::new();
    let mut key_pairs = vec![];
    let mut authority_key_pairs = vec![];
    let mut rng = StdRng::from_seed([0; 32]);

    for (i, stake) in authorities_stake.into_iter().enumerate() {
        let authority_keypair = AuthorityKeyPair::generate(&mut rng);
        let protocol_keypair = ProtocolKeyPair::generate(&mut rng);
        let network_keypair = NetworkKeyPair::generate(&mut rng);

        let name = AuthorityName::from(authority_keypair.public());

        authorities.insert(
            name,
            Authority {
                stake,
                address: get_available_local_address(),
                hostname: format!("test_host_{i}").to_string(),
                authority_key: authority_keypair.public().clone(),
                protocol_key: protocol_keypair.public(),
                network_key: network_keypair.public(),
            },
        );

        voting_weights.insert(name, stake);
        key_pairs.push((network_keypair, protocol_keypair));
        authority_key_pairs.push(authority_keypair);
    }

    let committee = Committee::new(epoch, voting_weights, authorities);
    (committee, key_pairs, authority_key_pairs)
}

/// Returns a local address with an ephemeral port.
pub fn get_available_local_address() -> Multiaddr {
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
