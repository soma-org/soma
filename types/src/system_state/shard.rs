use crate::metadata::DownloadMetadata;
use crate::object::ObjectRef;
use crate::report::Report;
use crate::shard_crypto::keys::EncoderPublicKey;
use serde::{Deserialize, Serialize};

use crate::{base::SomaAddress, committee::EpochId};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Shard {
    /// Metadata
    pub download_metadata: DownloadMetadata,
    /// Escrowed amount for the shard
    pub amount: u64,
    /// Epoch at which the shard was created
    pub created_epoch: EpochId,
    /// Address of the data submitter
    pub data_submitter: SomaAddress,
    /// (Optional) Object of Target the data submitter is attempting to hit
    pub target: Option<ObjectRef>,
    /// Winning encoder (defaults to None, set on shard completion)
    pub winning_encoder: Option<EncoderPublicKey>,
    /// Evaluation scores (defaults to None, set on shard completion)
    pub evaluation_scores: Option<u64>, // TODO: change this when score types are defined
    /// Target scores (defaults on None, set on shard completion)
    pub target_scores: Option<u64>, // TODO change this when score types are defined
    /// Winner's summary embedding (defaults to None, set on shard completion)
    pub summary_embedding: Option<ObjectRef>,
    /// Winner's sampled embedding - may become a new target (defaults to None, set on shard completion)
    pub sampled_embedding: Option<ObjectRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Embedding {
    /// Bytes for embedding
    pub embedding: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Target {
    /// None if system-generated, Some if user-submitted data bounty
    pub creator: Option<SomaAddress>,
    /// Reward amount for hitting the data target
    pub amount: u64,
    /// Epoch at which the target was created
    pub created_epoch: EpochId,
    /// Target Embedding object to be hit
    pub target_embedding: ObjectRef,
    /// Winning Shard (defaults to None, set when Shard that is aiming for target beats highest score)
    pub winning_shard: Option<ObjectRef>,
}
