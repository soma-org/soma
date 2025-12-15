use std::collections::HashMap;

use crate::checksum::Checksum;
use crate::committee::Epoch;
use crate::crypto::NetworkPublicKey;
use crate::digests::CheckpointDigest;
use crate::encoder_committee::CountUnit;
use crate::entropy::{CheckpointEntropy, CheckpointEntropyProof};
use crate::error::{ShardError, ShardResult, SharedError};
use crate::finality::FinalityProof;
use crate::metadata::{DownloadMetadata, Metadata, MtlsDownloadMetadataAPI, ObjectPath};
use crate::object::ObjectRef;
use crate::shard_crypto::keys::EncoderPublicKey;
use crate::system_state::shard::{Shard as ShardObject, Target};
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
/// CheckpointEntropy is derived from VDF(CheckpointDigest, iterations)
#[derive(Debug, Serialize, Deserialize)]
pub struct ShardEntropy {
    metadata: Metadata,
    entropy: CheckpointEntropy,
}

impl ShardEntropy {
    pub fn new(metadata: Metadata, entropy: CheckpointEntropy) -> Self {
        Self { metadata, entropy }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardAuthToken {
    pub finality_proof: FinalityProof,
    /// VDF output computed on checkpoint_digest
    pub checkpoint_entropy: CheckpointEntropy,
    pub checkpoint_entropy_proof: CheckpointEntropyProof,
    pub shard_ref: ObjectRef,
}

impl ShardAuthToken {
    pub fn new(
        finality_proof: FinalityProof,
        checkpoint_entropy: CheckpointEntropy,
        checkpoint_entropy_proof: CheckpointEntropyProof,
        shard_ref: ObjectRef,
    ) -> Self {
        Self {
            finality_proof,
            checkpoint_entropy,
            checkpoint_entropy_proof,
            shard_ref,
        }
    }

    pub fn finality_proof(&self) -> FinalityProof {
        self.finality_proof.clone()
    }

    pub fn checkpoint_entropy(&self) -> CheckpointEntropy {
        self.checkpoint_entropy.clone()
    }

    pub fn checkpoint_entropy_proof(&self) -> CheckpointEntropyProof {
        self.checkpoint_entropy_proof.clone()
    }

    pub fn shard_ref(&self) -> ObjectRef {
        self.shard_ref.clone()
    }

    pub fn checkpoint_digest(&self) -> &CheckpointDigest {
        self.finality_proof.checkpoint_digest()
    }

    pub fn epoch(&self) -> u64 {
        self.finality_proof.epoch()
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
    fn input_download_metadata(&self) -> &DownloadMetadata;
    fn target_embedding(&self) -> Option<Vec<u8>>;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputV1 {
    auth_token: ShardAuthToken,
    input_download_metadata: DownloadMetadata,
    shard_object: ShardObject,
    target_object: Option<Target>,
}

impl InputV1 {
    pub fn new(
        auth_token: ShardAuthToken,
        input_download_metadata: DownloadMetadata,
        shard_object: ShardObject,
        target_object: Option<Target>,
    ) -> Self {
        Self {
            auth_token,
            input_download_metadata,
            shard_object,
            target_object,
        }
    }
}

impl InputAPI for InputV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn input_download_metadata(&self) -> &DownloadMetadata {
        &self.input_download_metadata
    }
    fn target_embedding(&self) -> Option<Vec<u8>> {
        self.target_object.clone().map(|o| o.target_embedding)
    }
}

pub fn verify_input(input: &Input, shard: &Shard, peer: &NetworkPublicKey) -> SharedResult<()> {
    Ok(())
}

// /////////////////////////////////////////////////////
#[enum_dispatch]
pub trait GetDataAPI {
    fn object_paths(&self) -> &Vec<(ObjectPath, Metadata)>;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GetDataV1 {
    object_paths: Vec<(ObjectPath, Metadata)>,
}

impl GetDataV1 {
    pub fn new(object_paths: Vec<(ObjectPath, Metadata)>) -> Self {
        Self { object_paths }
    }
}

impl GetDataAPI for GetDataV1 {
    fn object_paths(&self) -> &Vec<(ObjectPath, Metadata)> {
        &self.object_paths
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[enum_dispatch(GetDataAPI)]
pub enum GetData {
    V1(GetDataV1),
}
// /////////////////////////////////////////////////////

#[enum_dispatch]
pub trait DownloadLocationsAPI {
    fn download_locations(&self) -> &HashMap<Checksum, DownloadMetadata>;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DownloadLocationsV1 {
    download_locations: HashMap<Checksum, DownloadMetadata>,
}

impl DownloadLocationsV1 {
    pub fn new(download_locations: HashMap<Checksum, DownloadMetadata>) -> Self {
        Self { download_locations }
    }
}

impl DownloadLocationsAPI for DownloadLocationsV1 {
    fn download_locations(&self) -> &HashMap<Checksum, DownloadMetadata> {
        &self.download_locations
    }
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[enum_dispatch(DownloadLocationsAPI)]
pub enum DownloadLocations {
    V1(DownloadLocationsV1),
}
