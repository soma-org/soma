use crate::metadata::MetadataCommitment;
use crate::shard_crypto::digest::Digest;
use serde::{Deserialize, Serialize};

use crate::{base::SomaAddress, committee::EpochId, submission::Submission};

/// ShardInput represents an escrowed amount for data encoding
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardInput {
    /// Metadata commitment digest - cryptographic identifier of the data
    pub digest: Digest<MetadataCommitment>,
    /// Size of data in bytes
    pub data_size_bytes: usize,
    /// Escrowed amount for the shard
    pub amount: u64,
    /// Epoch at which the shard expires
    pub expiration_epoch: EpochId,
    /// Address of the submitter
    pub submitter: SomaAddress,
}

/// Scores associated with a particular metadata commitment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardResult {
    /// Metadata commitment digest the scores are for
    pub digest: Digest<MetadataCommitment>,
    /// Size of data in bytes
    pub data_size_bytes: usize,
    /// Escrowed amount for the shard
    pub amount: u64,
    /// The winning submission
    pub submission: Submission,
}
