use std::{
    fmt::{Display, Formatter},
    ops::{Index, IndexMut},
};

use serde::{Deserialize, Serialize};

use crate::{
    crypto::keys::{AuthorityPublicKey, NetworkPublicKey, ProtocolPublicKey},
    multiaddr::Multiaddr,
};

/// The Committees [`AuthorityCommittee`] [`NetworkCommittee`][net] [`EncoderCommittee`][enc] are updated each epoch.
/// Epoch transitions are when all stake, registration, and reconfiguration related operations are applied.
///
/// [net]: crate::types::network_committee::NetworkCommittee
/// [enc]: crate::types::encoder_committee::EncoderCommittee
type Epoch = u64;

/// Voting power of a given committee member, roughly proportional to the actual amount of Soma staked
/// in a given member.
/// Total stake / voting power of all authorities should sum to 10,000.
//TODO: rename to voting power?
type Stake = u64;

/// Committee is the set of authorities that participate in the consensus protocol for this epoch.
/// Its configuration is stored and computed on chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthorityCommittee {
    /// The epoch number of this committee
    epoch: Epoch,
    /// Total stake in the committee.
    total_stake: Stake,
    /// The quorum threshold (2f+1).
    quorum_threshold: Stake,
    /// The validity threshold (f+1).
    validity_threshold: Stake,
    /// Protocol and network info of each authority.
    authorities: Vec<Authority>,
}

impl AuthorityCommittee {
    /// Creates a new [`AuthorityCommittee`] for a given Epoch and a vector of [`Authority`]
    fn new(epoch: Epoch, authorities: Vec<Authority>) -> Self {
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
            total_stake,
            quorum_threshold,
            validity_threshold,
            authorities,
        }
    }

    /// -----------------------------------------------------------------------
    /// Accessors to Committee fields.

    /// Returns the epoch field
    pub(crate) const fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// Returns the total stake field
    pub(crate) const fn total_stake(&self) -> Stake {
        self.total_stake
    }

    /// Returns quorum threhsold field
    pub(crate) const fn quorum_threshold(&self) -> Stake {
        self.quorum_threshold
    }

    /// Returns the validity threshold field
    pub(crate) const fn validity_threshold(&self) -> Stake {
        self.validity_threshold
    }

    /// Returns the stake for a given authority index
    pub(crate) fn stake(&self, authority_index: AuthorityIndex) -> Stake {
        self.authorities[authority_index].stake
    }

    /// Returns the authority for a given authority index
    pub(crate) fn authority(&self, authority_index: AuthorityIndex) -> &Authority {
        &self.authorities[authority_index]
    }

    /// Returns all the authorities as an iterator
    pub(crate) fn authorities(&self) -> impl Iterator<Item = (AuthorityIndex, &Authority)> {
        self.authorities
            .iter()
            .enumerate()
            .map(|(i, a)| (AuthorityIndex(i as u32), a))
    }

    /// -----------------------------------------------------------------------
    /// Helpers for Committee properties.

    /// Returns true if the provided stake has reached quorum (2f+1).
    pub(crate) const fn reached_quorum(&self, stake: Stake) -> bool {
        stake >= self.quorum_threshold()
    }

    /// Returns true if the provided stake has reached validity (f+1).
    pub(crate) const fn reached_validity(&self, stake: Stake) -> bool {
        stake >= self.validity_threshold()
    }

    /// Coverts an index to an [`AuthorityIndex`], if valid.
    /// Returns None if index is out of bound.
    pub(crate) fn to_authority_index(&self, index: usize) -> Option<AuthorityIndex> {
        if index < self.authorities.len() {
            Some(AuthorityIndex(index as u32))
        } else {
            None
        }
    }

    /// Returns true if the provided index is valid.
    pub(crate) fn is_valid_index(&self, index: AuthorityIndex) -> bool {
        index.value() < self.size()
    }

    /// Returns number of authorities in the committee.
    pub(crate) fn size(&self) -> usize {
        self.authorities.len()
    }
}

/// Represents one authority in the committee.
///
/// NOTE: this is intentionally un-cloneable, to encourage only copying relevant fields.
/// [`AuthorityIndex`] should be used to reference an authority instead.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Authority {
    /// Voting power of the authority in the committee.
    pub(crate) stake: Stake,
    /// Network address for communicating with the authority.
    pub(crate) address: Multiaddr,
    /// The authority's hostname, for metrics and logging.
    pub(crate) hostname: String,
    /// The authority's public key as Sui identity.
    //TODO: CHANGE THIS TO A CORRECT KEY
    pub(crate) authority_key: AuthorityPublicKey,
    /// The authority's public key for verifying blocks.
    pub(crate) protocol_key: ProtocolPublicKey,
    /// The authority's public key for TLS and as network identity.
    pub(crate) network_key: NetworkPublicKey,
}

/// Each authority is uniquely identified by its AuthorityIndex in the Committee.
/// AuthorityIndex is between 0 (inclusive) and the total number of authorities (exclusive).
///
/// NOTE: for safety, invalid AuthorityIndex should be impossible to create. So AuthorityIndex
/// should not be created or incremented outside of this file. AuthorityIndex received from peers
/// should be validated before use.
#[derive(
    Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug, Default, Hash, Serialize, Deserialize,
)]
pub(crate) struct AuthorityIndex(u32);

impl AuthorityIndex {
    /// Minimum committee size is 1, so 0 index is always valid.
    pub(crate) const ZERO: Self = Self(0);

    /// Only for scanning rows in the database. Invalid elsewhere.
    pub(crate) const MIN: Self = Self::ZERO;

    /// Only for scanning rows in the database. Invalid elsewhere.
    pub(crate) const MAX: Self = Self(u32::MAX);

    /// Converts the authority index to usize
    pub(crate) const fn value(&self) -> usize {
        self.0 as usize
    }
}

impl AuthorityIndex {
    /// creates a new index for testing
    #[cfg(test)]
    pub(crate) fn new_for_test(index: u32) -> Self {
        Self(index)
    }
}

// TODO: re-evaluate formats for production debugging.
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

impl<T, const N: usize> Index<AuthorityIndex> for [T; N] {
    type Output = T;

    fn index(&self, index: AuthorityIndex) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T> Index<AuthorityIndex> for Vec<T> {
    type Output = T;

    fn index(&self, index: AuthorityIndex) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T, const N: usize> IndexMut<AuthorityIndex> for [T; N] {
    fn index_mut(&mut self, index: AuthorityIndex) -> &mut Self::Output {
        self.get_mut(index.value()).unwrap()
    }
}

impl<T> IndexMut<AuthorityIndex> for Vec<T> {
    fn index_mut(&mut self, index: AuthorityIndex) -> &mut Self::Output {
        self.get_mut(index.value()).unwrap()
    }
}

pub struct AuthorityBitSet([u8; 32]); // 32 bytes = 256 authorities

impl AuthorityBitSet {
    pub fn new(authorities: &[AuthorityIndex]) -> Self {
        let mut bits = [0u8; 32];
        for idx in authorities {
            let byte_idx = idx.value() / 8;
            let bit_idx = idx.value() as usize % 8;
            bits[byte_idx] |= 1 << bit_idx;
        }
        Self(bits)
    }

    pub fn get_indices(&self) -> Vec<AuthorityIndex> {
        let mut authorities = Vec::new();
        for (byte_idx, &byte) in self.0.iter().enumerate() {
            for bit_idx in 0..8 {
                if byte & (1 << bit_idx) != 0 {
                    authorities.push(AuthorityIndex((byte_idx * 8 + bit_idx) as u32));
                }
            }
        }
        authorities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset() {
        let authorities = vec![
            AuthorityIndex::new_for_test(0),
            AuthorityIndex::new_for_test(7),
            AuthorityIndex::new_for_test(8),
            AuthorityIndex::new_for_test(254),
        ];

        let bitset = AuthorityBitSet::new(&authorities);

        // First byte should be 10000001
        assert_eq!(bitset.0[0], 0b10000001);
        // Second byte should be 00000001
        assert_eq!(bitset.0[1], 0b00000001);
        // Last byte should be 01000000
        assert_eq!(bitset.0[31], 0b01000000);
    }

    #[test]
    fn test_bitset_roundtrip() {
        let authorities = vec![
            AuthorityIndex::new_for_test(0),
            AuthorityIndex::new_for_test(7),
            AuthorityIndex::new_for_test(8),
            AuthorityIndex::new_for_test(254),
        ];

        let bitset = AuthorityBitSet::new(&authorities);
        let recovered = bitset.get_indices();
        assert_eq!(authorities, recovered);
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
