use bytes::Bytes;

use serde::{Deserialize, Serialize};

use crate::{
    checkpoints::CertifiedCheckpointSummary,
    committee::EpochId,
    consensus::{
        block::Round,
        commit::{CommitDigest, CommittedSubDag},
    },
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FetchCommitteesRequest {
    pub start: EpochId,
    pub end: EpochId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FetchCommitteesResponse {
    pub epoch_committees: Vec<CertifiedCheckpointSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetLatestEpochRequest {
    pub start: EpochId, // unused
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetLatestEpochResponse {
    pub epoch: EpochId,
}
