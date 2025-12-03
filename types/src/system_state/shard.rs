use crate::metadata::DownloadMetadata;
use crate::report::Report;
use serde::{Deserialize, Serialize};

use crate::{base::SomaAddress, committee::EpochId};

/// ShardInput represents an escrowed amount for data encoding
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardInput {
    /// Metadata
    pub download_metadata: DownloadMetadata,
    /// Escrowed amount for the shard
    pub amount: u64,
    /// Epoch at which the shard expires
    pub expiration_epoch: EpochId,
    /// Address of the submitter
    pub submitter: SomaAddress,
}

/// Scores associated with a particular metadata commitment
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ShardResult {
    /// Metadata
    pub download_metadata: DownloadMetadata,
    /// Escrowed amount for the shard
    pub amount: u64,
    /// The submitted report
    pub report: Report,
}
