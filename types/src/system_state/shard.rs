use crate::metadata::DownloadMetadata;
use crate::object::ObjectRef;
use crate::report::Report;
use crate::shard_crypto::keys::EncoderPublicKey;
use serde::{Deserialize, Serialize};

use crate::{base::SomaAddress, committee::EpochId};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Shard {
    /// Metadata for how to download the input data
    pub input_download_metadata: DownloadMetadata,
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
    /// Metadata for how to download the winner's complete embeddings (defaults to None, set on shard completion)
    pub embeddings_download_metadata: Option<DownloadMetadata>,
    /// Evaluation scores (defaults to None, set on shard completion)
    pub evaluation_scores: Option<u64>, // TODO: change this when score types are defined
    /// Target scores (defaults on None, set on shard completion)
    pub target_scores: Option<u64>, // TODO change this when score types are defined
    /// Winner's summary embedding (defaults to None, set on shard completion)
    pub summary_embedding: Option<Vec<u8>>,
    /// Winner's sampled embedding - may become a new target (defaults to None, set on shard completion)
    pub sampled_embedding: Option<Vec<u8>>,
}

/// Origin of a target determines reward source and refund behavior
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TargetOrigin {
    /// System-generated from shard's sampled embedding
    /// Reward comes from emissions pool based on epoch
    System,
    /// User-created bounty with escrowed reward
    /// If no winner, reward returns to creator
    User {
        creator: SomaAddress,
        reward_amount: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Target {
    /// Origin determines reward source and fallback recipient
    pub origin: TargetOrigin,
    /// Epoch at which the target was created
    pub created_epoch: EpochId,
    /// Target embedding to be matched
    pub target_embedding: Vec<u8>,
    /// Winner info stored directly (avoids loading shard during claim)
    pub winning_shard: Option<WinningShardInfo>,
}

impl Target {
    pub fn creator(&self) -> Option<SomaAddress> {
        match &self.origin {
            TargetOrigin::System => None,
            TargetOrigin::User { creator, .. } => Some(*creator),
        }
    }

    pub fn is_user_created(&self) -> bool {
        matches!(self.origin, TargetOrigin::User { .. })
    }
}

/// Info about the winning shard - stored in Target to avoid loading shard object
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct WinningShardInfo {
    pub shard_ref: ObjectRef,
    pub data_submitter: SomaAddress,
    pub winning_encoder: EncoderPublicKey,
    pub distance: u64,
    pub shard_created_epoch: EpochId,
}
