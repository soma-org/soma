use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::{EncoderKeyPair, EncoderPublicKey, NetworkKeyPair, NetworkPublicKey}, digest::Digest, multiaddr::Multiaddr
};
use std::{
    fmt::{Display, Formatter},
    ops::{Index, IndexMut},
};
use rand::{rngs::StdRng, seq::index::sample_weighted, SeedableRng};


use crate::error::{ShardError, ShardResult};

use super::shard::{Shard, ShardEntropy};

/// max of 10_000 
type VotingPowerUnit = u16;
/// Size of a shard, must not be larger than shard index size hence u32
type ShardSizeUnit = u32;
/// Count of nodes, valid between 1 and shard size
type QuorumUnit = u32;
/// Epoch associated with the committee
type Epoch = u64;

/// Holds a single encoder committee for a given modality. Each modality has a unique set of
/// Encoders. A given encoder can register to multiple modalities, but are not required to
/// register to all encoders. Additionally, stake is normalized to 10_000, making it extremely
/// important to ensure that encoders from one modality cannot mix with others.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EncoderCommittee {
    /// committee changes with epoch
    epoch: Epoch,
    /// current shard size requirement
    shard_size: ShardSizeUnit,
    /// the number required for quorum (can change with epoch)
    quorum_threshold: QuorumUnit,
    /// all the encoders
    encoders: Vec<Encoder>,
}

impl EncoderCommittee {
    /// creates a new encoder committee for a given modality marker
    fn new(
        epoch: Epoch,
        encoders: Vec<Encoder>,
        shard_size: ShardSizeUnit,
        quorum_threshold: QuorumUnit,
    ) -> Self {
        assert!(!encoders.is_empty(), "Committee cannot be empty!");
        assert!(
            encoders.len() < u32::MAX as usize,
            "Too many encoders ({})!",
            encoders.len()
        );

        // let total_stake = encoders.iter().map(|a| a.stake).sum();
        // assert_ne!(total_stake, 0, "Total stake cannot be zero!");
        Self {
            epoch,
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
    // pub(crate) fn total_stake(&self) -> Stake {
    //     self.total_stake
    // }

    /// returns selection threshold
    pub(crate) fn shard_size(&self) -> ShardSizeUnit {
        self.shard_size
    }
    /// returns quorum threshold
    pub(crate) fn quorum_threshold(&self) -> QuorumUnit {
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
    pub(crate) fn reached_quorum(&self, count: QuorumUnit) -> bool {
        count as u32 >= self.quorum_threshold()
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
        
        let weight_fn = |index: usize| -> f64 {
            let encoder_index = EncoderIndex(index as u32);
            self.voting_power(encoder_index) as f64
        };
        
        let index_vec = sample_weighted(&mut rng, self.size(), weight_fn, self.shard_size as usize).map_err(|e| ShardError::WeightedSampleError(e.to_string()))?;
        let encoders = index_vec.into_iter().map(|index| EncoderIndex(index as u32)).collect();

        Ok(Shard::new(self.epoch, self.quorum_threshold, encoders))
    }
}

#[cfg(test)]
impl EncoderCommittee {
    pub fn local_test_committee(
        epoch: Epoch,
        voting_powers: Vec<VotingPowerUnit>,
        shard_size: ShardSizeUnit,
        quorum_threshold: QuorumUnit,
        starting_port: u16,
    ) -> (Self, Vec<(NetworkKeyPair, EncoderKeyPair)>) {
        let mut rng = StdRng::from_seed([0; 32]);
        let mut key_pairs = vec![];
        let encoders = voting_powers
            .into_iter()
            .enumerate()
            .map(|(i, power)| {
                let encoder_keypair = EncoderKeyPair::generate(&mut rng);
                let network_keypair = NetworkKeyPair::generate(&mut rng);
                let port = starting_port + i as u16;
                
                key_pairs.push((network_keypair.clone(), encoder_keypair.clone()));
                
                Encoder {
                    voting_power: power,
                    address: format!("/ip4/127.0.0.1/tcp/{}", port).parse().unwrap(),
                    hostname: format!("test-encoder-{}", i),
                    encoder_key: encoder_keypair.public(),
                    network_key: network_keypair.public(),
                }
            })
            .collect();

        (Self::new(epoch, encoders, shard_size, quorum_threshold), key_pairs)
    }
}

/// Holds all the data for a given Encoder modality
// TODO: switch to arc'ing these details to make the code more efficient if the same encoder
// is a member of multiple modalities
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Encoder {
    /// Voting power of the authority in the committee.
    voting_power: VotingPowerUnit,
    /// Network address for communicating with the authority.
    address: Multiaddr,
    /// The authority's hostname, for metrics and logging.
    hostname: String,
    /// The authority's public key as Sui identity.
    encoder_key: EncoderPublicKey,
    /// The authority's public key for TLS and as network identity.
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

    // const fn new(index: u32) -> Self {
    //     Self(index)
    // }
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



#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use shared::digest::Digest;

    #[test]
    fn test_sample_shard() {
        // Create a committee with different voting powers
        let voting_powers = vec![100, 200, 300, 400, 500];
        let shard_size = 3;
        let quorum_threshold = 2;
        let starting_port = 8000;
        let epoch = 1;

        let (committee, _keys) = EncoderCommittee::local_test_committee(
            epoch,
            voting_powers,
            shard_size,
            quorum_threshold,
            starting_port,
        );

        // Sample a shard using test entropy
        let entropy = Digest::<ShardEntropy>::new_from_bytes(&Bytes::from("test_entropy".as_bytes().to_vec()));
        let shard = committee.sample_shard(entropy).unwrap();

        // Verify shard properties
        assert_eq!(shard.epoch(), epoch);
        assert_eq!(shard.quorum_threshold(), quorum_threshold);
        assert_eq!(shard.size(), shard_size as usize);

        // Verify all indices are valid
        for encoder_index in shard.encoders() {
            assert!(committee.is_valid_index(encoder_index));
        }
    }

    #[test]
    fn test_sample_shard_weighted_distribution() {
        let voting_powers = vec![1000, 100, 100, 100, 100];  // First encoder has 10x voting power
        let shard_size = 2;
        let quorum_threshold = 2;
        let starting_port = 8000;
        let epoch = 1;
        let trials = 1000;
        
        let (committee, _keys) = EncoderCommittee::local_test_committee(
            epoch,
            voting_powers.clone(),
            shard_size,
            quorum_threshold,
            starting_port,
        );

        let mut selection_counts = vec![0; voting_powers.len()];
        
        for i in 0..trials {
            let entropy = Digest::<ShardEntropy>::new_from_bytes(&Bytes::from(format!("test_entropy_{}", i).as_bytes().to_vec()));
            let shard = committee.sample_shard(entropy).unwrap();
            
            for encoder_index in shard.encoders() {
                selection_counts[encoder_index.value()] += 1;
            }
        }

        println!("{:?}", selection_counts);

        // The first encoder (with 10x voting power) should be selected significantly more often
        assert!(selection_counts[0] > selection_counts[1] * 2);
    }
}