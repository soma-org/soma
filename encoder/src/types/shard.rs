//! In a previous version, the shard was composed of a single set of encoders.
//! What was realized is that the security (probability of a dishonest majority)
//! should be scaled seperately from the number of computers performing computation.
//! This allows for the computation set that is generating an embedding to be tuned
//! independently of security considerations. The seperation of concerns is also slightly
//! more secure compared to encoders that are directly impacted by the outcome.
use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::EncoderPublicKey, digest::Digest, entropy::BlockEntropy,
    metadata::MetadataCommitment,
};

use crate::error::{ShardError, ShardResult};

use super::encoder_committee::{CountUnit, EvaluationEncoder, InferenceEncoder};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) enum ShardRole {
    Inference(InferenceEncoder),
    Evaluation(EvaluationEncoder),
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Shard {
    minimum_inference_size: CountUnit,
    evaluation_quorum_threshold: CountUnit,
    inference_set: Vec<InferenceEncoder>,
    evaluation_set: Vec<EvaluationEncoder>,
    seed: Digest<ShardEntropy>,
}

impl Shard {
    pub(crate) fn inference_set_contains(&self, inference_encoder: &InferenceEncoder) -> bool {
        self.inference_set.contains(inference_encoder)
    }

    pub(crate) fn evaluation_set_contains(&self, evaluation_encoder: &EvaluationEncoder) -> bool {
        self.evaluation_set.contains(evaluation_encoder)
    }
    pub(crate) fn inference_set_size(&self) -> usize {
        self.inference_set.len()
    }

    pub(crate) fn evaluation_set_size(&self) -> usize {
        self.evaluation_set.len()
    }
    pub(crate) fn minimum_inference_size(&self) -> CountUnit {
        self.minimum_inference_size
    }
    pub(crate) fn evaluation_quorum_threshold(&self) -> CountUnit {
        self.evaluation_quorum_threshold
    }
    pub(crate) fn role(&self, encoder: EncoderPublicKey) -> ShardResult<ShardRole> {
        let ie = InferenceEncoder::new(encoder.clone());
        let ee = EvaluationEncoder::new(encoder);

        if self.inference_set_contains(&ie) {
            return Ok(ShardRole::Inference(ie));
        } else if self.evaluation_set_contains(&ee) {
            return Ok(ShardRole::Evaluation(ee));
        }
        Err(ShardError::InvalidShardMember)
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
    entropy: BlockEntropy,
}

impl ShardEntropy {
    pub fn new(metadata_commitment: MetadataCommitment, entropy: BlockEntropy) -> Self {
        Self {
            metadata_commitment,
            entropy,
        }
    }
}
