// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::{AuthorityPublicKey, NetworkPublicKey, ProtocolPublicKey},
    multiaddr::Multiaddr,
};
use std::{
    fmt::{Display, Formatter},
    ops::{Index, IndexMut},
};

/// Count represents a count of shard members
type Count = usize;
type Epoch = u64;
type Stake = u64;

/// Holds a single encoder committee for a given modality. Each modality has a unique set of
/// Encoders. A given encoder can register to multiple modalities, but are not required to
/// register to all encoders. Additionally, stake is normalized to 10_000, making it extremely
/// important to ensure that encoders from one modality cannot mix with others.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EncoderCommittee {
    /// committee changes with epoch
    epoch: Epoch,
    /// total_stake of the committee
    total_stake: Stake,
    /// current shard size requirement
    shard_size: Count,
    /// the number required for quorum (can change with epoch)
    quorum_threshold: Count,
    /// all the encoders
    encoders: Vec<Encoder>,
}

impl EncoderCommittee {
    /// creates a new encoder committee for a given modality marker
    fn new(
        epoch: Epoch,
        encoders: Vec<Encoder>,
        shard_size: Count,
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
            shard_size,
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
    pub(crate) fn shard_size(&self) -> Count {
        self.shard_size
    }
    /// returns quorum threshold
    pub(crate) fn quorum_threshold(&self) -> Count {
        self.quorum_threshold
    }

    /// returns stake for a given encoder index (of the specific modality)
    pub(crate) fn stake(&self, encoder_index: EncoderIndex) -> Stake {
        self.encoders[encoder_index].stake
    }

    /// returns the encoder at a specified encoder index
    pub(crate) fn encoder(&self, encoder_index: EncoderIndex) -> &Encoder {
        &self.encoders[encoder_index]
    }

    /// returns all the encoders
    pub(crate) fn encoders(&self) -> impl Iterator<Item = (EncoderIndex, &Encoder)> {
        self.encoders
            .iter()
            .enumerate()
            .map(|(i, a)| (EncoderIndex(i as u32), a))
    }

    /// Returns true if the provided stake has reached quorum (2f+1).
    pub(crate) fn reached_quorum(&self, count: Count) -> bool {
        count >= self.quorum_threshold()
    }

    /// Coverts an index to an EncoderIndex, if valid.
    /// Returns None if index is out of bound.
    pub(crate) fn to_encoder_index(&self, index: usize) -> Option<EncoderIndex> {
        if index < self.encoders.len() {
            Some(EncoderIndex(index as u32))
        } else {
            None
        }
    }

    /// Returns true if the provided index is valid.
    pub(crate) fn is_valid_index(&self, index: EncoderIndex) -> bool {
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
pub(crate) struct Encoder {
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
}

/// Represents an EncoderIndex, also modality marked for type safety
#[derive(
    Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug, Default, Hash, Serialize, Deserialize,
)]
pub(crate) struct EncoderIndex(u32);

impl EncoderIndex {
    /// Minimum committee size is 1, so 0 index is always valid.
    const ZERO: Self = Self(0);

    /// Only for scanning rows in the database. Invalid elsewhere.
    const MIN: Self = Self::ZERO;
    /// Max lex for scanning rows
    const MAX: Self = Self(u32::MAX);

    /// returns the value
    const fn value(&self) -> usize {
        self.0 as usize
    }

    const fn new(index: u32) -> Self {
        Self(index)
    }
}

#[cfg(test)]
impl EncoderIndex {
    /// creates an encoder index of specific modality for tests only
    const fn new_for_test(index: u32) -> Self {
        Self(index)
    }
}

// TODO: re-evaluate formats for production debugging.
impl Display for EncoderIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.value() < 26 {
            let c = (b'A' + self.value() as u8) as char;
            f.write_str(c.to_string().as_str())
        } else {
            write!(f, "[{:02}]", self.value())
        }
    }
}

impl<T, const N: usize> Index<EncoderIndex> for [T; N] {
    type Output = T;

    fn index(&self, index: EncoderIndex) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T> Index<EncoderIndex> for Vec<T> {
    type Output = T;

    fn index(&self, index: EncoderIndex) -> &Self::Output {
        self.get(index.value()).unwrap()
    }
}

impl<T, const N: usize> IndexMut<EncoderIndex> for [T; N] {
    fn index_mut(&mut self, index: EncoderIndex) -> &mut Self::Output {
        self.get_mut(index.value()).unwrap()
    }
}

impl<T> IndexMut<EncoderIndex> for Vec<T> {
    fn index_mut(&mut self, index: EncoderIndex) -> &mut Self::Output {
        self.get_mut(index.value()).unwrap()
    }
}
