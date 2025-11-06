use crate::committee::Epoch;
use crate::crypto::NetworkPublicKey;
use crate::encoder_committee::CountUnit;
use crate::entropy::{BlockEntropy, BlockEntropyProof};
use crate::error::{ShardError, ShardResult, SharedError};
use crate::finality::FinalityProof;
use crate::metadata::{DownloadMetadata, Metadata, MtlsDownloadMetadataAPI};
use crate::object::ObjectRef;
use crate::shard_crypto::keys::EncoderPublicKey;
use crate::{error::SharedResult, shard_crypto::digest::Digest};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Shard {
    quorum_threshold: CountUnit,
    encoders: Vec<EncoderPublicKey>,
    pub seed: Digest<ShardEntropy>,
    epoch: Epoch,
}

impl Shard {
    pub fn new(
        quorum_threshold: CountUnit,
        mut encoders: Vec<EncoderPublicKey>,
        seed: Digest<ShardEntropy>,
        epoch: Epoch,
    ) -> Self {
        // ensure the same encoder order
        encoders.sort();
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
    metadata: Metadata,
    entropy: BlockEntropy,
}

impl ShardEntropy {
    pub fn new(metadata: Metadata, entropy: BlockEntropy) -> Self {
        Self { metadata, entropy }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardAuthToken {
    pub finality_proof: FinalityProof,
    pub block_entropy: BlockEntropy,
    pub block_entropy_proof: BlockEntropyProof,
    pub metadata: Metadata,
    pub shard_input_ref: ObjectRef,
}

impl ShardAuthToken {
    pub fn new(
        finality_proof: FinalityProof,
        block_entropy: BlockEntropy,
        block_entropy_proof: BlockEntropyProof,
        metadata: Metadata,
        shard_input_ref: ObjectRef,
    ) -> Self {
        Self {
            finality_proof,
            block_entropy,
            block_entropy_proof,
            metadata,
            shard_input_ref,
        }
    }

    pub fn finality_proof(&self) -> FinalityProof {
        self.finality_proof.clone()
    }

    pub fn block_entropy(&self) -> BlockEntropy {
        self.block_entropy.clone()
    }

    pub fn block_entropy_proof(&self) -> BlockEntropyProof {
        self.block_entropy_proof.clone()
    }

    pub fn metadata(&self) -> Metadata {
        self.metadata.clone()
    }

    pub fn shard_input_ref(&self) -> ObjectRef {
        self.shard_input_ref.clone()
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
    fn download_metadata(&self) -> &DownloadMetadata;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputV1 {
    auth_token: ShardAuthToken,
    download_metadata: DownloadMetadata,
}

impl InputV1 {
    pub fn new(auth_token: ShardAuthToken, download_metadata: DownloadMetadata) -> Self {
        Self {
            auth_token,
            download_metadata,
        }
    }
}

impl InputAPI for InputV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn download_metadata(&self) -> &DownloadMetadata {
        &self.download_metadata
    }
}

pub fn verify_input(input: &Input, shard: &Shard, peer: &NetworkPublicKey) -> SharedResult<()> {
    match input.download_metadata() {
        DownloadMetadata::Mtls(download_metadata) => {
            if download_metadata.peer() != peer {
                return Err(SharedError::FailedTypeVerification(
                    "sending peer must match tls key in input".to_string(),
                ));
            }
        }
        _ => {}
    }
    Ok(())
}
