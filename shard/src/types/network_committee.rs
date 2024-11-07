use std::{
    fmt::{Display, Formatter},
    ops::{Index, IndexMut},
};

use serde::{Deserialize, Serialize};

use crate::crypto::keys::{AuthorityPublicKey, NetworkPublicKey, ProtocolPublicKey};
use crate::types::multiaddr::Multiaddr;

use crate::types::authority_committee::{Epoch, Stake};

/// The network identities that meet the minimum required amount of stake
/// to communicate with other nodes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkCommittee {
    /// The epoch number of this committee
    epoch: Epoch,
    /// Protocol and network info of each authorized authority.
    identities: Vec<NetworkIdentity>,
}

impl NetworkCommittee {
    /// Creates a new network committee for a given epoch and vector of network identities
    pub fn new(epoch: Epoch, identities: Vec<NetworkIdentity>) -> Self {
        assert!(!identities.is_empty(), "Committee cannot be empty!");
        assert!(
            identities.len() < u32::MAX as usize,
            "Too many authorities ({})!",
            identities.len()
        );

        assert!(
            !identities.is_empty(),
            "No identities meet the minimum stake requirement!"
        );

        Self { epoch, identities }
    }
    /// -----------------------------------------------------------------------
    /// Accessors to Committee fields.

    pub const fn epoch(&self) -> Epoch {
        self.epoch
    }

    // pub fn total_stake(&self) -> Stake {
    //     self.total_stake
    // }

    // pub fn minimum_stake(&self) -> Stake {
    //     self.minimum_stake
    // }

    /// returns the identitity for a given network index
    pub fn identity(&self, index: NetworkingIndex) -> &NetworkIdentity {
        &self.identities[index]
    }

    /// returns all the network identities as an iterator
    pub fn identities(&self) -> impl Iterator<Item = (NetworkingIndex, &NetworkIdentity)> {
        self.identities
            .iter()
            .enumerate()
            .map(|(i, a)| (NetworkingIndex(i as u32), a))
    }

    /// -----------------------------------------------------------------------
    /// Helpers for Committee properties.

    /// Coverts an index to an NetworkingIndex, if valid.
    /// Returns None if index is out of bound.
    pub fn to_identity_index(&self, index: usize) -> Option<NetworkingIndex> {
        if index < self.identities.len() {
            Some(NetworkingIndex(index as u32))
        } else {
            None
        }
    }

    /// Returns true if the provided index is valid.
    pub fn is_valid_index(&self, index: NetworkingIndex) -> bool {
        index.value() < self.size()
    }

    /// Returns number of authorities in the committee.
    pub fn size(&self) -> usize {
        self.identities.len()
    }
}

/// Network identitiy, this should likely be reused in multiple places.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkIdentity {
    /// Network address for communicating with the authority.
    pub address: Multiaddr,
    /// The authority's hostname, for metrics and logging.
    pub hostname: String,
    /// The authority's public key for verifying blocks.
    pub protocol_key: ProtocolPublicKey,
    /// The authority's public key for TLS and as network identity.
    pub network_key: NetworkPublicKey,
}

/// Each authority is uniquely identified by its `NetworkingIndex` in the Committee.
/// `NetworkingIndex` is between 0 (inclusive) and the total number of authorities (exclusive).
///
/// NOTE: for safety, invalid `NetworkingIndex` should be impossible to create. So `NetworkingIndex`
/// should not be created or incremented outside of this file. `NetworkingIndex` received from peers
/// should be validated before use.
#[derive(
    Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug, Default, Hash, Serialize, Deserialize,
)]
pub struct NetworkingIndex(u32);

impl NetworkingIndex {
    /// Minimum committee size is 1, so 0 index is always valid.
    pub const ZERO: Self = Self(0);

    /// Only for scanning rows in the database. Invalid elsewhere.
    pub const MIN: Self = Self::ZERO;
    /// Only for scanning rows in the database. Invalid elsewhere.
    pub const MAX: Self = Self(u32::MAX);

    /// converts to usize
    pub const fn value(self) -> usize {
        self.0 as usize
    }
}

#[cfg(test)]
impl NetworkingIndex {
    pub const fn new_for_test(index: u32) -> Self {
        Self(index)
    }
}

// TODO: re-evaluate formats for production debugging.
impl Display for NetworkingIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.value() < 26 {
            let c = (b'A' + self.value() as u8) as char;
            f.write_str(c.to_string().as_str())
        } else {
            write!(f, "[{:02}]", self.value())
        }
    }
}

impl<T, const N: usize> Index<NetworkingIndex> for [T; N] {
    type Output = T;

    fn index(&self, index: NetworkingIndex) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T> Index<NetworkingIndex> for Vec<T> {
    type Output = T;

    fn index(&self, index: NetworkingIndex) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T, const N: usize> IndexMut<NetworkingIndex> for [T; N] {
    fn index_mut(&mut self, index: NetworkingIndex) -> &mut Self::Output {
        self.get_mut(index.value()).unwrap()
    }
}

impl<T> IndexMut<NetworkingIndex> for Vec<T> {
    fn index_mut(&mut self, index: NetworkingIndex) -> &mut Self::Output {
        self.get_mut(index.value()).unwrap()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::local_committee_and_keys;

//     #[test]
//     fn committee_basic() {
//         // GIVEN
//         let epoch = 100;
//         let num_of_authorities = 9;
//         let authority_stakes = (1..=9).map(|s| s as Stake).collect();
//         let (committee, _) = local_committee_and_keys(epoch, authority_stakes);

//         // THEN make sure the output Committee fields are populated correctly.
//         assert_eq!(committee.size(), num_of_authorities);
//         for (i, authority) in committee.authorities() {
//             assert_eq!((i.value() + 1) as Stake, authority.stake);
//         }

//         // AND ensure thresholds are calculated correctly.
//         assert_eq!(committee.total_stake(), 45);
//         assert_eq!(committee.quorum_threshold(), 31);
//         assert_eq!(committee.validity_threshold(), 15);
//     }
// }
