use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use shared::{digest::Digest, metadata::MetadataCommitment};

use crate::{base::SomaAddress, committee::EpochId};

/// ShardInput represents an escrowed amount for data encoding
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardInput {
    /// Metadata commitment digest - cryptographic identifier of the data
    pub digest: Digest<MetadataCommitment>,
    /// Size of data in bytes
    pub data_size_bytes: u64,
    /// Escrowed amount for the shard
    pub amount: u64,
    /// Epoch at which the shard expires
    pub expiration_epoch: EpochId,
    /// Address of the submitter
    pub submitter: SomaAddress,
}

/// Scores associated with a particular metadata commitment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScoreSet {
    /// Metadata commitment digest the scores are for
    pub digest: Digest<MetadataCommitment>,
    /// Size of data in bytes
    pub data_size_bytes: u64,
    /// Score entries mapping encoder addresses to their scores
    pub scores: BTreeMap<SomaAddress, u64>,
    /// Total score across all encoders
    pub total_score: u64,
    /// Epoch when the scores were reported
    pub reported_epoch: EpochId,
}
// TODO: agg sig, ShardToken?, ScoreSet?, shard_ref (Digest<Shard?)?
