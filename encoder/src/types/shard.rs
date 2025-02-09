use std::hash::{Hash, Hasher};

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::{digest::Digest, entropy::BlockEntropyOutput, metadata::MetadataCommitment};
use strum_macros::Display;

use super::encoder_committee::{CountUnit, EncoderIndex, Epoch};

pub(crate) struct Shard {
    epoch: Epoch,
    minimum_inference_size: CountUnit,
    evaluation_quorum_threshold: CountUnit,
    inference_set: Vec<EncoderIndex>,
    evaluation_set: Vec<EncoderIndex>,
}

impl Shard {
    pub(crate) fn new(
        epoch: Epoch,
        minimum_inference_size: CountUnit,
        evaluation_quorum_threshold: CountUnit,
        inference_set: Vec<EncoderIndex>,
        evaluation_set: Vec<EncoderIndex>,
    ) -> Self {
        Self {
            epoch,
            minimum_inference_size,
            evaluation_quorum_threshold,
            inference_set,
            evaluation_set,
        }
    }
    pub(crate) fn epoch(&self) -> Epoch {
        self.epoch
    }
    pub(crate) fn inference_set(&self) -> Vec<EncoderIndex> {
        self.inference_set.clone()
    }

    pub(crate) fn evaluation_set(&self) -> Vec<EncoderIndex> {
        self.evaluation_set.clone()
    }
    pub(crate) fn inference_size(&self) -> usize {
        self.inference_set.len()
    }

    pub(crate) fn evaluation_size(&self) -> usize {
        self.evaluation_set.len()
    }
    pub(crate) fn minimum_inference_size(&self) -> CountUnit {
        self.minimum_inference_size
    }
    pub(crate) fn evaluation_quorum_threshold(&self) -> CountUnit {
        self.evaluation_quorum_threshold
    }

    // #[cfg(test)]
    // pub(crate) fn new_for_test(
    //     epoch: Epoch,
    //     quorum_threshold: QuorumUnit,
    //     encoders: Vec<EncoderIndex>,
    // ) -> Self {
    //     Self::new(epoch, quorum_threshold, encoders)
    // }
}

/// The Digest<ShardEntropy> acts as a seed for random sampling from the encoder committee.
/// Digest<MetadataCommitment> is included inside of a tx which is a one way fn whereas
/// this entropy uses the actual values of the serialized type of MetadataCommitment to create the Digest.
///
/// BlockEntropy is derived from VDF(Epoch, BlockRef, iterations)
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ShardEntropy {
    metadata_commitment: MetadataCommitment,
    entropy: BlockEntropyOutput,
}

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Display)]
#[enum_dispatch(ShardRefAPI)]
pub enum ShardRef {
    V1(ShardRefV1),
}

/// `ShardRefAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
trait ShardRefAPI {
    fn epoch(&self) -> &Epoch;
    fn seed(&self) -> &Digest<ShardEntropy>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct ShardRefV1 {
    /// the epoch that this shard was sampled from, important since committees change each epoch
    epoch: Epoch,
    /// the digest from the tbls threshold signature and data hash that when combined forms a unique source of randomness
    seed: Digest<ShardEntropy>,
}

impl ShardRefV1 {
    /// create a shard commit v1
    pub(crate) const fn new(epoch: Epoch, seed: Digest<ShardEntropy>) -> Self {
        Self { epoch, seed }
    }
}

impl ShardRefAPI for ShardRefV1 {
    fn epoch(&self) -> &Epoch {
        &self.epoch
    }

    fn seed(&self) -> &Digest<ShardEntropy> {
        &self.seed
    }
}

// impl ShardRef {
//     /// lex min.
//     const MIN: Self = Self {
//         epoch: 0,
//         leader: NetworkingIndex::MIN,
//         entropy_digest: Digest::MIN,
//         modality: Modality::text(),
//     };

//     /// lex max
//     const MAX: Self = Self {
//         epoch: u64::MAX,
//         leader: NetworkingIndex::MAX,
//         entropy_digest: Digest::MAX,
//         modality: Modality::video(),
//     };

// }

impl Hash for ShardRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.seed().as_ref()[..8]);
    }
}
