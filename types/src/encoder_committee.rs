//! The encoder committee stores all the neccessary details for creating shards from staked nodes
//! The set type specifies whether to sample the total shard (inference and evaluation) at once or
//! whether it should be resampled. In all cases it should opt for disjoint sets except for the cases
//! where the size of staked encoders does not allow it.
use crate::{
    checksum::Checksum,
    committee::{EpochId, VotingPower},
    crypto::{
        AuthorityKeyPair, AuthorityPublicKey, AuthoritySignature,
        DefaultHash as DefaultHashFunction, NetworkPublicKey, DIGEST_LENGTH,
    },
    error::{ConsensusError, ConsensusResult},
    intent::{Intent, IntentMessage, IntentScope},
    multiaddr::Multiaddr,
    shard::{Shard, ShardEntropy},
    shard_crypto::{digest::Digest, keys::EncoderPublicKey},
};
use fastcrypto::{
    hash::HashFunction,
    traits::{Signer, VerifyingKey},
};
use rand::{rngs::StdRng, seq::index::sample_weighted, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fmt::{Display, Formatter},
    ops::{Index, IndexMut},
};

use crate::error::{ShardError, ShardResult};

/// Count of nodes, valid between 1 and shard size
pub(crate) type CountUnit = u32;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct EncoderCommittee {
    /// The epoch number
    pub epoch: EpochId,

    /// The active encoders
    encoders: BTreeMap<EncoderPublicKey, Encoder>,

    shard_size: CountUnit,
    quorum_threshold: CountUnit,

    /// Quick lookup index
    index_map: BTreeMap<EncoderPublicKey, usize>,

    /// Reverse mapping for index -> key lookup
    // key_by_index: Vec<EncoderPublicKey>,

    /// Network metadata for encoders
    pub network_metadata: BTreeMap<EncoderPublicKey, EncoderNetworkMetadata>,
}

impl EncoderCommittee {
    /// Primary constructor that builds both index mappings
    pub fn new(
        epoch: EpochId,
        encoders: Vec<Encoder>,
        shard_size: CountUnit,
        quorum_threshold: CountUnit,
        network_metadata: BTreeMap<EncoderPublicKey, EncoderNetworkMetadata>,
    ) -> Self {
        // Build the BTreeMap and index mappings
        let mut encoder_map = BTreeMap::new();
        let mut index_map = BTreeMap::new();
        // let mut key_by_index = Vec::new();

        // Sort encoders by key for consistent ordering
        let mut sorted_encoders = encoders;
        sorted_encoders.sort_by_key(|e| e.encoder_key.clone());

        for (index, encoder) in sorted_encoders.into_iter().enumerate() {
            let key = encoder.encoder_key.clone();
            index_map.insert(key.clone(), index);
            // key_by_index.push(key.clone());
            encoder_map.insert(key, encoder);
        }

        Self {
            epoch,
            encoders: encoder_map,
            shard_size,
            quorum_threshold,
            index_map,
            // key_by_index,
            network_metadata,
        }
    }

    // pub fn encoder(&self, encoder_index: EncoderIndex) -> Option<&Encoder> {
    //     let idx: usize = encoder_index.into();
    //     self.key_by_index
    //         .get(idx)
    //         .and_then(|key| self.encoders.get(key))
    // }

    pub fn encoder_by_key(&self, key: &EncoderPublicKey) -> Option<&Encoder> {
        self.encoders.get(key)
    }

    // pub fn voting_power(&self, encoder_index: EncoderIndex) -> Option<u64> {
    //     self.encoder(encoder_index).map(|e| e.voting_power)
    // }

    pub fn voting_power_by_key(&self, key: &EncoderPublicKey) -> Option<u64> {
        self.encoders.get(key).map(|e| e.voting_power)
    }

    // pub fn encoders(&self) -> impl Iterator<Item = (EncoderIndex, &Encoder)> + '_ {
    //     self.key_by_index
    //         .iter()
    //         .enumerate()
    //         .filter_map(move |(idx, key)| {
    //             self.encoders
    //                 .get(key)
    //                 .map(|encoder| (EncoderIndex::new(idx as u32), encoder))
    //         })
    // }

    pub fn to_encoder_index(&self, key: &EncoderPublicKey) -> Option<EncoderIndex> {
        self.index_map
            .get(key)
            .map(|&idx| EncoderIndex::new(idx as u32))
    }

    // pub fn is_valid_index(&self, index: EncoderIndex) -> bool {
    //     let idx: usize = index.into();
    //     idx < self.key_by_index.len()
    // }

    pub fn members(&self) -> BTreeMap<EncoderPublicKey, u64> {
        self.encoders
            .iter()
            .map(|(key, encoder)| (key.clone(), encoder.voting_power))
            .collect()
    }

    pub fn contains(&self, key: &EncoderPublicKey) -> bool {
        self.encoders.contains_key(key)
    }

    pub fn size(&self) -> usize {
        self.encoders.len()
    }

    pub fn shard_size(&self) -> CountUnit {
        self.shard_size
    }

    pub fn quorum_threshold(&self) -> CountUnit {
        self.quorum_threshold
    }

    pub fn reached_quorum(&self, count: CountUnit) -> bool {
        count >= self.quorum_threshold
    }

    pub fn sample_shard(&self, entropy: Digest<ShardEntropy>) -> ShardResult<Shard> {
        let mut rng = StdRng::from_seed(entropy.into());

        // Collect all encoders with their weights
        let encoder_vec: Vec<(&EncoderPublicKey, &Encoder)> = self.encoders.iter().collect();

        // Weight function using voting power
        let weight_fn = |index: usize| -> f64 { encoder_vec[index].1.voting_power as f64 };

        // Sample encoders
        let encoder_pubkeys = sample_weighted(
            &mut rng,
            encoder_vec.len(),
            weight_fn,
            self.shard_size as usize,
        )
        .map_err(|e| ShardError::WeightedSampleError(e.to_string()))?
        .into_iter()
        .map(|index| encoder_vec[index].0.clone())
        .collect::<Vec<_>>();

        Ok(Shard::new(
            self.quorum_threshold,
            encoder_pubkeys,
            entropy,
            self.epoch,
        ))
    }

    pub fn compute_digest(&self) -> ConsensusResult<EncoderCommitteeDigest> {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(bcs::to_bytes(self).map_err(ConsensusError::SerializationFailure)?);
        Ok(EncoderCommitteeDigest(hasher.finalize().into()))
    }

    pub fn sign(&self, keypair: &AuthorityKeyPair) -> ConsensusResult<AuthoritySignature> {
        let digest = self.compute_digest()?;
        let message = bcs::to_bytes(&to_encoder_committee_intent(digest))
            .map_err(ConsensusError::SerializationFailure)?;
        Ok(keypair.sign(&message))
    }

    pub fn verify_signature(
        &self,
        signature: &AuthoritySignature,
        public_key: &AuthorityPublicKey,
    ) -> ConsensusResult<()> {
        let digest = self.compute_digest()?;
        let message = bcs::to_bytes(&to_encoder_committee_intent(digest))
            .map_err(ConsensusError::SerializationFailure)?;
        public_key
            .verify(&message, signature)
            .map_err(ConsensusError::SignatureVerificationFailure)
    }
}

// impl EncoderCommittee {
//     /// creates a new encoder committee for a given modality marker
//     pub fn new(
//         epoch: Epoch,
//         shard_size: CountUnit,
//         quorum_threshold: CountUnit,
//         encoders: Vec<Encoder>,
//     ) -> Self {
//         assert!(!encoders.is_empty(), "Committee cannot be empty!");
//         assert!(
//             encoders.len() < u32::MAX as usize,
//             "Too many encoders ({})!",
//             encoders.len()
//         );

//         assert!(
//             encoders.len() >= shard_size as usize,
//             "len of encoders must be greater or equal to evaluation set size"
//         );

//         assert!(
//             shard_size >= quorum_threshold,
//             "evaluation set size must be greater or equal to quorum size"
//         );

//         // let mut encoders = encoders;
//         // encoders.sort_by_key(|e| e.encoder_key.clone());

//         Self {
//             epoch,
//             shard_size,
//             quorum_threshold,
//             encoders,
//         }
//     }

//     // -----------------------------------------------------------------------

//     /// returns the epoch
//     pub(crate) fn epoch(&self) -> Epoch {
//         self.epoch
//     }
//     pub(crate) fn shard_size(&self) -> CountUnit {
//         self.shard_size
//     }
//     pub fn quorum_threshold(&self) -> CountUnit {
//         self.quorum_threshold
//     }

//     /// returns voting power for a given encoder index (of the specific modality)
//     pub(crate) fn voting_power(&self, encoder_index: EncoderIndex) -> VotingPowerUnit {
//         self.encoders[encoder_index].voting_power
//     }

//     /// returns the encoder at a specified encoder index
//     pub fn encoder(&self, encoder_index: EncoderIndex) -> &Encoder {
//         &self.encoders[encoder_index]
//     }

//     pub fn encoder_by_key(&self, encoder: &EncoderPublicKey) -> Option<&Encoder> {
//         self.encoders.iter().find(|e| &e.encoder_key == encoder)
//     }
//     /// returns all the encoders
//     pub(crate) fn encoders(&self) -> impl Iterator<Item = (EncoderIndex, &Encoder)> {
//         self.encoders
//             .iter()
//             .enumerate()
//             .map(|(i, a)| (EncoderIndex(i as u32), a))
//     }

//     /// Returns true if the provided count has reached quorum (2f+1).
//     pub(crate) fn reached_quorum(&self, count: CountUnit) -> bool {
//         count >= self.quorum_threshold()
//     }

//     /// Coverts an index to an EncoderIndex, if valid.
//     /// Returns None if index is out of bound.
//     pub(crate) fn to_encoder_index(&self, index: usize) -> Option<EncoderIndex> {
//         if index < self.encoders.len() {
//             Some(EncoderIndex(index as u32))
//         } else {
//             None
//         }
//     }

//     /// Returns true if the provided index is valid.
//     pub(crate) fn is_valid_index(&self, index: EncoderIndex) -> bool {
//         index.value() < self.size()
//     }

//     /// Returns number of authorities in the committee.
//     pub fn size(&self) -> usize {
//         self.encoders.len()
//     }

//     pub fn sample_shard(&self, entropy: Digest<ShardEntropy>) -> ShardResult<Shard> {
//         let mut rng = StdRng::from_seed(entropy.into());

//         // TODO: only compute this once per committee rather than every sample
//         let weight_fn = |index: usize| -> f64 {
//             let encoder_index = EncoderIndex(index as u32);
//             self.voting_power(encoder_index) as f64
//         };

//         let encoder_pubkeys =
//             sample_weighted(&mut rng, self.size(), weight_fn, self.shard_size() as usize)
//                 .map_err(|e| ShardError::WeightedSampleError(e.to_string()))?
//                 .into_iter()
//                 .map(|index| self.encoder(EncoderIndex(index as u32)).encoder_key.clone())
//                 .collect::<Vec<_>>();

//         Ok(Shard::new(
//             self.quorum_threshold,
//             encoder_pubkeys,
//             entropy,
//             self.epoch,
//         ))
//     }
// }

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq, Clone)]
pub struct EncoderNetworkMetadata {
    pub internal_network_address: Multiaddr,
    pub external_network_address: Multiaddr,
    pub object_server_address: Multiaddr,
    pub network_key: NetworkPublicKey,
    pub hostname: String,
}

/// Digest of encoder committee, used for signing
#[derive(Serialize, Deserialize)]
pub struct EncoderCommitteeDigest([u8; DIGEST_LENGTH]);

/// Wrap an EncoderCommitteeDigest in the intent message
pub fn to_encoder_committee_intent(
    digest: EncoderCommitteeDigest,
) -> IntentMessage<EncoderCommitteeDigest> {
    IntentMessage::new(Intent::consensus_app(IntentScope::EncoderCommittee), digest)
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Encoder {
    pub voting_power: VotingPower,
    pub encoder_key: EncoderPublicKey,
    pub probe_checksum: Checksum,
}

/// Represents an EncoderIndex, also modality marked for type safety
#[derive(
    Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug, Default, Hash, Serialize, Deserialize,
)]
pub struct EncoderIndex(u32);

impl EncoderIndex {
    /// Minimum committee size is 1, so 0 index is always valid.
    pub(crate) const ZERO: Self = Self(0);

    /// Only for scanning rows in the database. Invalid elsewhere.
    pub(crate) const MIN: Self = Self::ZERO;
    /// Max lex for scanning rows
    pub(crate) const MAX: Self = Self(u32::MAX);

    /// returns the value
    const fn value(&self) -> usize {
        self.0 as usize
    }

    pub fn new(index: u32) -> Self {
        Self(index)
    }
}

#[cfg(test)]
impl EncoderIndex {
    /// creates an encoder index of specific modality for tests only
    pub(crate) const fn new_for_test(index: u32) -> Self {
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

impl From<usize> for EncoderIndex {
    fn from(value: usize) -> Self {
        Self(value as u32)
    }
}

impl From<EncoderIndex> for usize {
    fn from(index: EncoderIndex) -> Self {
        index.value()
    }
}
