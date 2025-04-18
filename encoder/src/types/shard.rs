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

use super::encoder_committee::{CountUnit, Epoch};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Shard {
    quorum_threshold: CountUnit,
    encoders: Vec<EncoderPublicKey>,
    seed: Digest<ShardEntropy>,
    epoch: Epoch,
}

impl Shard {
    pub(crate) fn new(
        quorum_threshold: CountUnit,
        encoders: Vec<EncoderPublicKey>,
        seed: Digest<ShardEntropy>,
        epoch: Epoch,
    ) -> Self {
        Self {
            quorum_threshold,
            encoders,
            seed,
            epoch,
        }
    }
    pub(crate) fn encoders(&self) -> Vec<EncoderPublicKey> {
        self.encoders.clone()
    }
    pub(crate) fn size(&self) -> usize {
        self.encoders.len()
    }
    pub(crate) fn contains(&self, encoder: &EncoderPublicKey) -> bool {
        self.encoders.contains(encoder)
    }

    pub(crate) fn quorum_threshold(&self) -> CountUnit {
        self.quorum_threshold
    }

    pub(crate) fn digest(&self) -> ShardResult<Digest<Self>> {
        Digest::new(self).map_err(ShardError::DigestFailure)
    }
    pub(crate) fn epoch(&self) -> Epoch {
        self.epoch
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
