//! In a previous version, the shard was composed of a single set of encoders.
//! What was realized is that the security (probability of a dishonest majority)
//! should be scaled seperately from the number of computers performing computation.
//! This allows for the computation set that is generating an embedding to be tuned
//! independently of security considerations. The seperation of concerns is also slightly
//! more secure compared to encoders that are directly impacted by the outcome.
use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
};

use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::EncoderPublicKey, digest::Digest, entropy::BlockEntropyOutput,
    metadata::MetadataCommitment,
};

use super::encoder_committee::{CountUnit, EncoderIndex};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Shard {
    minimum_inference_size: CountUnit,
    evaluation_quorum_threshold: CountUnit,
    inference_set: Vec<EncoderPublicKey>,
    evaluation_set: Vec<EncoderPublicKey>,
    shard_ref: ShardRef,
}

impl Shard {
    // pub(crate) fn inference_set(&self) -> Vec<EncoderIndex> {
    //     self.inference_set.clone()
    // }
    pub(crate) fn inference_set_contains(&self, encoder: &EncoderPublicKey) -> bool {
        self.inference_set.contains(encoder)
    }

    pub(crate) fn evaluation_set_contains(&self, encoder: &EncoderPublicKey) -> bool {
        self.evaluation_set.contains(encoder)
    }
    // pub(crate) fn evaluation_set(&self) -> Vec<EncoderIndex> {
    //     self.evaluation_set.clone()
    // }
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

    pub(crate) fn contains(&self, encoder: &EncoderPublicKey) -> bool {
        self.inference_set.contains(encoder) || self.evaluation_set.contains(encoder)
    }

    pub(crate) fn shard_set(&self) -> Vec<EncoderPublicKey> {
        let mut peers_set: HashSet<EncoderPublicKey> =
            self.inference_set.clone().into_iter().collect();
        peers_set.extend(self.evaluation_set.clone());
        peers_set.into_iter().collect()
    }
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

impl ShardEntropy {
    pub fn new(metadata_commitment: MetadataCommitment, entropy: BlockEntropyOutput) -> Self {
        Self {
            metadata_commitment,
            entropy,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct ShardRef(Digest<ShardEntropy>);

impl Hash for ShardRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.as_ref()[..8]);
    }
}
