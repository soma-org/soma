use crate::committee::Epoch;
use crate::encoder_committee::CountUnit;
use crate::entropy::BlockEntropy;
use crate::error::{ShardError, ShardResult};
use crate::finality::FinalityProof;
use crate::metadata::MetadataCommitment;
use crate::shard_crypto::keys::EncoderPublicKey;
use crate::{error::SharedResult, shard_crypto::digest::Digest};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Shard {
    quorum_threshold: CountUnit,
    encoders: Vec<EncoderPublicKey>,
    seed: Digest<ShardEntropy>,
    epoch: Epoch,
}

impl Shard {
    pub fn new(
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
    pub fn encoders(&self) -> Vec<EncoderPublicKey> {
        self.encoders.clone()
    }
    pub fn size(&self) -> usize {
        self.encoders.len()
    }

    pub fn contains(&self, encoder: &EncoderPublicKey) -> bool {
        self.encoders.contains(encoder)
    }

    pub fn quorum_threshold(&self) -> CountUnit {
        self.quorum_threshold
    }

    pub fn rejection_threshold(&self) -> CountUnit {
        self.size() as u32 - self.quorum_threshold + 1
    }

    pub fn digest(&self) -> ShardResult<Digest<Self>> {
        Digest::new(self).map_err(ShardError::DigestFailure)
    }
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }
}

/// The Digest<ShardEntropy> acts as a seed for random sampling from the encoder committee.
/// Digest<MetadataCommitment> is included inside of a tx which is a one way fn whereas
/// this entropy uses the actual values of the serialized type of MetadataCommitment to create the Digest.
///
/// BlockEntropy is derived from VDF(Epoch, BlockRef, iterations)
#[derive(Debug, Serialize, Deserialize)]
pub struct ShardEntropy {
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardAuthToken {
    pub finality_proof: FinalityProof,
    pub metadata_commitment: MetadataCommitment,
    pub shard: Shard,
}

impl ShardAuthToken {
    pub fn new(
        finality_proof: FinalityProof,
        metadata_commitment: MetadataCommitment,
        shard: Shard,
    ) -> Self {
        Self {
            finality_proof,
            metadata_commitment,
            shard,
        }
    }
    pub fn metadata_commitment(&self) -> MetadataCommitment {
        self.metadata_commitment.clone()
    }

    pub fn epoch(&self) -> u64 {
        self.finality_proof.consensus_finality.leader_block.epoch
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[enum_dispatch(InputAPI)]
pub enum Input {
    V1(InputV1),
}

#[enum_dispatch]
pub trait InputAPI {
    fn auth_token(&self) -> &ShardAuthToken;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputV1 {
    auth_token: ShardAuthToken,
}

impl InputV1 {
    pub fn new(auth_token: ShardAuthToken) -> Self {
        Self { auth_token }
    }
}

impl InputAPI for InputV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
}

pub fn verify_input(input: &Input, shard: &Shard) -> SharedResult<()> {
    // TODO: need to fix this to work with the correct signature
    // input.verify_signature(Scope::Input, .author().inner())?;
    Ok(())
}
