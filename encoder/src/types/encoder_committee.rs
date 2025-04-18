//! The encoder committee stores all the neccessary details for creating shards from staked nodes
//! The set type specifies whether to sample the total shard (inference and evaluation) at once or
//! whether it should be resampled. In all cases it should opt for disjoint sets except for the cases
//! where the size of staked encoders does not allow it.
use rand::{rngs::StdRng, seq::index::sample_weighted, SeedableRng};
use serde::{Deserialize, Serialize};
use shared::{crypto::keys::EncoderPublicKey, digest::Digest, probe::ProbeMetadata};
use std::{
    fmt::{Display, Formatter},
    ops::{Index, IndexMut},
};

use crate::error::{ShardError, ShardResult};

use super::shard::{Shard, ShardEntropy};

/// max of 10_000
type VotingPowerUnit = u16;
/// Count of nodes, valid between 1 and shard size
pub(crate) type CountUnit = u32;
/// Epoch associated with the committee
pub(crate) type Epoch = u64;

/// Holds a single encoder committee for a given modality. Each modality has a unique set of
/// Encoders. A given encoder can register to multiple modalities, but are not required to
/// register to all encoders. Additionally, stake is normalized to 10_000, making it extremely
/// important to ensure that encoders from one modality cannot mix with others.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EncoderCommittee {
    /// committee changes with epoch
    epoch: Epoch,
    shard_size: CountUnit,
    quorum_threshold: CountUnit,
    encoders: Vec<Encoder>,
}

impl EncoderCommittee {
    /// creates a new encoder committee for a given modality marker
    fn new(
        epoch: Epoch,
        shard_size: CountUnit,
        quorum_threshold: CountUnit,
        encoders: Vec<Encoder>,
    ) -> Self {
        assert!(!encoders.is_empty(), "Committee cannot be empty!");
        assert!(
            encoders.len() < u32::MAX as usize,
            "Too many encoders ({})!",
            encoders.len()
        );

        assert!(
            encoders.len() >= shard_size as usize,
            "len of encoders must be greater or equal to evaluation set size"
        );

        assert!(
            shard_size >= quorum_threshold,
            "evaluation set size must be greater or equal to quorum size"
        );

        Self {
            epoch,
            shard_size,
            quorum_threshold,
            encoders,
        }
    }

    // -----------------------------------------------------------------------

    /// returns the epoch
    pub(crate) fn epoch(&self) -> Epoch {
        self.epoch
    }
    pub(crate) fn shard_size(&self) -> CountUnit {
        self.shard_size
    }
    pub(crate) fn quorum_threshold(&self) -> CountUnit {
        self.quorum_threshold
    }

    /// returns voting power for a given encoder index (of the specific modality)
    pub(crate) fn voting_power(&self, encoder_index: EncoderIndex) -> VotingPowerUnit {
        self.encoders[encoder_index].voting_power
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

    /// Returns true if the provided count has reached quorum (2f+1).
    pub(crate) fn reached_quorum(&self, count: CountUnit) -> bool {
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

    pub(crate) fn sample_shard(&self, entropy: Digest<ShardEntropy>) -> ShardResult<Shard> {
        let mut rng = StdRng::from_seed(entropy.into());

        // TODO: only compute this once per committee rather than every sample
        let weight_fn = |index: usize| -> f64 {
            let encoder_index = EncoderIndex(index as u32);
            self.voting_power(encoder_index) as f64
        };

        let encoder_pubkeys =
            sample_weighted(&mut rng, self.size(), weight_fn, self.shard_size() as usize)
                .map_err(|e| ShardError::WeightedSampleError(e.to_string()))?
                .into_iter()
                .map(|index| self.encoder(EncoderIndex(index as u32)).encoder_key.clone())
                .collect::<Vec<_>>();

        Ok(Shard::new(
            self.quorum_threshold,
            encoder_pubkeys,
            entropy,
            self.epoch,
        ))
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Encoder {
    pub voting_power: VotingPowerUnit,
    pub encoder_key: EncoderPublicKey,
    pub probe: ProbeMetadata,
}

/// Represents an EncoderIndex, also modality marked for type safety
#[derive(
    Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Debug, Default, Hash, Serialize, Deserialize,
)]
pub(crate) struct EncoderIndex(u32);

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

    // const fn new(index: u32) -> Self {
    //     Self(index)
    // }
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

// #[cfg(test)]
// impl EncoderCommittee {
//     pub fn local_test_committee(
//         epoch: Epoch,
//         encoder_details: Vec<(VotingPowerUnit, ProbeMetadata, EncoderStatus)>,
//         inference_set_size: CountUnit,
//         minimum_inference_size: CountUnit,
//         evaluation_set_size: CountUnit,
//         evaluation_quorum_threshold: CountUnit,
//         starting_port: u16,
//     ) -> (Self, Vec<(PeerKeyPair, EncoderKeyPair)>) {
//         let mut rng = StdRng::from_seed([0; 32]);
//         let mut key_pairs = vec![];
//         let encoders = encoder_details
//             .into_iter()
//             .enumerate()
//             .map(|(i, (power, probe, status))| {
//                 let encoder_keypair = EncoderKeyPair::generate(&mut rng);
//                 let peer_keypair = PeerKeyPair::generate(&mut rng);
//                 let port = starting_port + i as u16;

//                 key_pairs.push((peer_keypair.clone(), encoder_keypair.clone()));

//                 Encoder {
//                     voting_power: power,
//                     encoder_key: encoder_keypair.public(),
//                     probe,
//                     peer: peer_keypair.public(),
//                     status,
//                 }
//             })
//             .collect();

//         (
//             Self::new(
//                 epoch,
//                 inference_set_size,
//                 minimum_inference_size,
//                 evaluation_set_size,
//                 evaluation_quorum_threshold,
//                 encoders,
//             ),
//             key_pairs,
//         )
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::collections::{HashMap, HashSet};

//     // Helper function to generate probe digests
//     fn probe(i: u8) -> ProbeMetadata {
//         ProbeMetadata::new_for_test(&[i; 32])
//     }

//     #[test]
//     fn test_committee_sampling_safety() {
//         // Test with exact size to force intersecting sets
//         let encoder_details = vec![
//             (100, probe(0), EncoderStatus::Active),
//             (100, probe(1), EncoderStatus::Active),
//             (100, probe(2), EncoderStatus::Active),
//         ];

//         let (committee, _) = EncoderCommittee::local_test_committee(
//             1,
//             encoder_details,
//             2, // inference_size
//             1, // min_inference
//             2, // eval_size
//             1, // quorum
//             8000,
//         );

//         assert!(matches!(committee.set_type, SetType::Intersecting));

//         // Sample multiple times to ensure safety properties
//         for seed in 0..100 {
//             let shard = committee
//                 .sample_shard(Digest::new_from_bytes([seed; 32]))
//                 .unwrap();

//             // Basic size checks
//             assert_eq!(shard.inference_set().len(), 2);
//             assert_eq!(shard.evaluation_set().len(), 2);
//             assert_eq!(shard.epoch(), committee.epoch());

//             let all_indices = [shard.inference_set(), shard.evaluation_set()].concat();
//             // All indices must be valid
//             for idx in all_indices {
//                 assert!(committee.is_valid_index(idx));
//                 assert!(idx.value() < committee.size());
//             }
//         }
//     }

//     #[test]
//     fn test_disjoint_sampling_safety() {
//         // Test with larger size to force disjoint sets
//         let encoder_details = vec![
//             (1000, probe(0), EncoderStatus::Active), // High power
//             (500, probe(1), EncoderStatus::Active),
//             (250, probe(2), EncoderStatus::Active),
//             (125, probe(3), EncoderStatus::Active),
//             (60, probe(4), EncoderStatus::Active),
//             (30, probe(5), EncoderStatus::Active), // Low power
//         ];

//         let (committee, _) = EncoderCommittee::local_test_committee(
//             1,
//             encoder_details,
//             2, // inference_size
//             1, // min_inference
//             2, // eval_size
//             1, // quorum
//             8000,
//         );

//         assert!(matches!(committee.set_type, SetType::Disjoint(4)));

//         // Track frequency of selection for each index
//         let mut frequencies = HashMap::new();
//         let samples = u8::MAX;

//         for seed in 0..samples {
//             let shard = committee
//                 .sample_shard(Digest::new_from_bytes([seed; 32]))
//                 .unwrap();

//             // Sets must be disjoint
//             let inference_set: HashSet<_> = shard.inference_set().iter().cloned().collect();
//             let eval_set: HashSet<_> = shard.evaluation_set().iter().cloned().collect();
//             assert!(
//                 inference_set.is_disjoint(&eval_set),
//                 "Sets must be disjoint"
//             );

//             // Track frequencies
//             for idx in inference_set.iter().chain(eval_set.iter()) {
//                 *frequencies.entry(idx.value()).or_insert(0) += 1;
//             }

//             // Verify shard properties
//             assert_eq!(shard.inference_set().len(), 2);
//             assert_eq!(shard.evaluation_set().len(), 2);
//         }

//         // Higher voting power nodes should be selected more frequently
//         assert!(
//             frequencies.get(&0).unwrap_or(&0) > frequencies.get(&5).unwrap_or(&0),
//             "Higher power nodes should be selected more often"
//         );
//     }

//     #[test]
//     fn test_weighted_sampling_distribution() {
//         // Test extreme voting power differences
//         let encoder_details = vec![
//             (10000, probe(0), EncoderStatus::Active), // Extremely high power
//             (1, probe(1), EncoderStatus::Active),     // Minimal power
//             (1, probe(2), EncoderStatus::Active),     // Minimal power
//             (1, probe(3), EncoderStatus::Active),     // Minimal power
//         ];

//         let (committee, _) = EncoderCommittee::local_test_committee(
//             1,
//             encoder_details,
//             1, // inference_size
//             1, // min_inference
//             1, // eval_size
//             1, // quorum
//             8000,
//         );

//         let mut high_power_selections = 0;
//         let samples = u8::MAX;

//         for seed in 0..samples {
//             let shard = committee
//                 .sample_shard(Digest::new_from_bytes([seed; 32]))
//                 .unwrap();
//             let all_indices = [shard.inference_set(), shard.evaluation_set()].concat();

//             if all_indices.contains(&EncoderIndex(0)) {
//                 high_power_selections += 1;
//             }
//         }

//         // With 10000:1 power ratio, the high power node should be selected almost always
//         assert!(
//             high_power_selections as f64 / samples as f64 > 0.95,
//             "High power node selected only {}/{} times",
//             high_power_selections,
//             samples
//         );
//     }

//     #[test]
//     fn test_minimum_inference_threshold() {
//         let encoder_details = vec![(100, probe(0)), (100, probe(1)), (100, probe(2))];

//         let (committee, _) = EncoderCommittee::local_test_committee(
//             1,
//             encoder_details,
//             2, // inference_size
//             2, // min_inference - equal to inference_size
//             1, // eval_size
//             1, // quorum
//             8000,
//         );

//         for seed in 0..100 {
//             let shard = committee
//                 .sample_shard(Digest::new_from_bytes([seed; 32]))
//                 .unwrap();
//             assert_eq!(
//                 shard.inference_set().len(),
//                 committee.minimum_inference_size() as usize,
//                 "Must have exactly minimum inference nodes"
//             );
//         }
//     }

//     #[test]
//     fn test_quorum_properties() {
//         let encoder_details = vec![
//             (100, probe(0), EncoderStatus::Active),
//             (100, probe(1), EncoderStatus::Active),
//             (100, probe(2), EncoderStatus::Active),
//             (100, probe(3), EncoderStatus::Active),
//         ];

//         let (committee, _) = EncoderCommittee::local_test_committee(
//             1,
//             encoder_details,
//             2, // inference_size
//             1, // min_inference
//             3, // eval_size
//             2, // quorum - requires 2/3 evaluators
//             8000,
//         );

//         assert!(!committee.reached_quorum(1), "1/3 should not reach quorum");
//         assert!(committee.reached_quorum(2), "2/3 should reach quorum");
//         assert!(committee.reached_quorum(3), "3/3 should reach quorum");

//         for seed in 0..100 {
//             let shard = committee
//                 .sample_shard(Digest::new_from_bytes([seed; 32]))
//                 .unwrap();
//             assert!(
//                 shard.evaluation_set().len() >= committee.evaluation_quorum_threshold() as usize,
//                 "Must have enough evaluators to potentially reach quorum"
//             );
//         }
//     }

//     #[test]
//     #[should_panic(expected = "evaluation set size must be greater or equal to quorum size")]
//     fn test_invalid_quorum_config() {
//         let encoder_details = vec![
//             (100, probe(0), EncoderStatus::Active),
//             (100, probe(1), EncoderStatus::Active),
//         ];

//         EncoderCommittee::local_test_committee(
//             1,
//             encoder_details,
//             1, // inference_size
//             1, // min_inference
//             1, // eval_size
//             2, // quorum larger than eval_size - should panic
//             8000,
//         );
//     }

//     #[test]
//     fn test_epoch_consistency() {
//         let encoder_details = vec![
//             (100, probe(0), EncoderStatus::Active),
//             (100, probe(1), EncoderStatus::Active),
//         ];

//         let epoch = 42;
//         let (committee, _) = EncoderCommittee::local_test_committee(
//             epoch,
//             encoder_details,
//             1, // inference_size
//             1, // min_inference
//             1, // eval_size
//             1, // quorum
//             8000,
//         );

//         assert_eq!(committee.epoch(), epoch);

//         for seed in 0..10 {
//             let shard = committee
//                 .sample_shard(Digest::new_from_bytes([seed; 32]))
//                 .unwrap();
//             assert_eq!(
//                 shard.epoch(),
//                 epoch,
//                 "Shard epoch must match committee epoch"
//             );
//         }
//     }
// }
