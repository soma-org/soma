// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::crypto::keys::{AuthorityPublicKey, NetworkPublicKey, ProtocolPublicKey};
use crate::types::authority_committee::Epoch;
use crate::types::authority_committee::Stake;
use crate::types::multiaddr::Multiaddr;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{
    fmt::{Display, Formatter},
    ops::{Index, IndexMut},
};

use super::modality::ModalityMarker;

/// Count represents a count of shard members
type Count = usize;

/// Holds a single encoder committee for a given modality. Each modality has a unique set of
/// Encoders. A given encoder can register to multiple modalities, but are not required to
/// register to all encoders. Additionally, stake is normalized to 10_000, making it extremely
/// important to ensure that encoders from one modality cannot mix with others.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EncoderCommittee<M: ModalityMarker> {
    /// committee changes with epoch
    epoch: Epoch,
    /// total_stake of the committee
    total_stake: Stake,
    /// the number required for commit selection (can change with epoch)
    selection_threshold: Count,
    /// the number required for quorum (can change with epoch)
    quorum_threshold: Count,
    /// all the encoders
    encoders: Vec<Encoder<M>>,
}

impl<M: ModalityMarker> EncoderCommittee<M> {
    /// creates a new encoder committee for a given modality marker
    fn new(
        epoch: Epoch,
        encoders: Vec<Encoder<M>>,
        selection_threshold: Count,
        quorum_threshold: Count,
    ) -> Self {
        assert!(!encoders.is_empty(), "Committee cannot be empty!");
        assert!(
            encoders.len() < u32::MAX as usize,
            "Too many encoders ({})!",
            encoders.len()
        );

        let total_stake = encoders.iter().map(|a| a.stake).sum();
        assert_ne!(total_stake, 0, "Total stake cannot be zero!");
        Self {
            epoch,
            total_stake,
            selection_threshold,
            quorum_threshold,
            encoders,
        }
    }

    /// -----------------------------------------------------------------------
    /// Accessors to Committee fields.

    /// returns the epoch
    pub(crate) fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// returns total stake
    pub(crate) fn total_stake(&self) -> Stake {
        self.total_stake
    }

    /// returns selection threshold
    pub(crate) fn selection_threshold(&self) -> Count {
        self.selection_threshold
    }
    /// returns quorum threshold
    pub(crate) fn quorum_threshold(&self) -> Count {
        self.quorum_threshold
    }

    /// returns stake for a given encoder index (of the specific modality)
    pub(crate) fn stake(&self, encoder_index: EncoderIndex<M>) -> Stake {
        self.encoders[encoder_index].stake
    }

    /// returns the encoder at a specified encoder index
    pub(crate) fn encoder(&self, encoder_index: EncoderIndex<M>) -> &Encoder<M> {
        &self.encoders[encoder_index]
    }

    /// returns all the encoders
    pub(crate) fn encoders(&self) -> impl Iterator<Item = (EncoderIndex<M>, &Encoder<M>)> {
        self.encoders
            .iter()
            .enumerate()
            .map(|(i, a)| (EncoderIndex(i as u32, PhantomData), a))
    }

    /// -----------------------------------------------------------------------
    /// Helpers for Committee properties.

    /// Returns true if the provided stake has reached validity (f+1).
    pub(crate) fn reached_selection(&self, count: Count) -> bool {
        count >= self.selection_threshold()
    }
    /// Returns true if the provided stake has reached quorum (2f+1).
    pub(crate) fn reached_quorum(&self, count: Count) -> bool {
        count >= self.quorum_threshold()
    }

    /// Coverts an index to an EncoderIndex, if valid.
    /// Returns None if index is out of bound.
    pub(crate) fn to_encoder_index(&self, index: usize) -> Option<EncoderIndex<M>> {
        if index < self.encoders.len() {
            Some(EncoderIndex(index as u32, PhantomData))
        } else {
            None
        }
    }

    /// Returns true if the provided index is valid.
    pub(crate) fn is_valid_index(&self, index: EncoderIndex<M>) -> bool {
        index.value() < self.size()
    }

    /// Returns number of authorities in the committee.
    pub(crate) fn size(&self) -> usize {
        self.encoders.len()
    }
}

/// Holds all the data for a given Encoder modality
// TODO: switch to arc'ing these details to make the code more efficient if the same encoder
// is a member of multiple modalities
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Encoder<M: ModalityMarker> {
    /// Voting power of the authority in the committee.
    stake: Stake,
    /// Network address for communicating with the authority.
    address: Multiaddr,
    /// The authority's hostname, for metrics and logging.
    hostname: String,
    /// The authority's lic key as Sui identity.
    authority_key: AuthorityPublicKey,
    /// The authority's lic key for verifying blocks.
    protocol_key: ProtocolPublicKey,
    /// The authority's lic key for TLS and as network identity.
    network_key: NetworkPublicKey,
    /// phantom data to enable the static type checking with modality marker
    _marker: PhantomData<M>,
}

/// Represents an EncoderIndex, also modality marked for type safety
#[derive(
    Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug, Default, Hash, Serialize, Deserialize,
)]
pub(crate) struct EncoderIndex<M: ModalityMarker>(u32, PhantomData<M>);

impl<M: ModalityMarker> EncoderIndex<M> {
    /// Minimum committee size is 1, so 0 index is always valid.
    const ZERO: Self = Self(0, PhantomData);

    /// Only for scanning rows in the database. Invalid elsewhere.
    const MIN: Self = Self::ZERO;
    /// Max lex for scanning rows
    const MAX: Self = Self(u32::MAX, PhantomData);

    /// returns the value
    const fn value(&self) -> usize {
        self.0 as usize
    }

    const fn new(index: u32) -> Self {
        Self(index, PhantomData)
    }
}

#[cfg(test)]
impl<M: ModalityMarker> EncoderIndex<M> {
    /// creates an encoder index of specific modality for tests only
    const fn new_for_test(index: u32) -> Self {
        Self(index, PhantomData)
    }
}

// TODO: re-evaluate formats for production debugging.
impl<M: ModalityMarker> Display for EncoderIndex<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.value() < 26 {
            let c = (b'A' + self.value() as u8) as char;
            f.write_str(c.to_string().as_str())
        } else {
            write!(f, "[{:02}]", self.value())
        }
    }
}

impl<T, const N: usize, M: ModalityMarker> Index<EncoderIndex<M>> for [T; N] {
    type Output = T;

    fn index(&self, index: EncoderIndex<M>) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T, M: ModalityMarker> Index<EncoderIndex<M>> for Vec<T> {
    type Output = T;

    fn index(&self, index: EncoderIndex<M>) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T, const N: usize, M: ModalityMarker> IndexMut<EncoderIndex<M>> for [T; N] {
    fn index_mut(&mut self, index: EncoderIndex<M>) -> &mut Self::Output {
        self.get_mut(index.value()).unwrap()
    }
}

impl<T, M: ModalityMarker> IndexMut<EncoderIndex<M>> for Vec<T> {
    fn index_mut(&mut self, index: EncoderIndex<M>) -> &mut Self::Output {
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
