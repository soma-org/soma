use bytes::Bytes;

use serde::{Deserialize, Serialize};

use crate::{
    accumulator::CommitIndex,
    committee::EpochId,
    consensus::{
        block::Round,
        commit::{CommitDigest, CommittedSubDag},
    },
};

#[derive(Clone, prost::Message, Serialize, Deserialize)]
pub struct FetchCommitteesRequest {
    #[prost(uint64, tag = "1")]
    pub start: EpochId,
    #[prost(uint64, tag = "2")]
    pub end: EpochId,
}

#[derive(Clone, prost::Message, Serialize, Deserialize)]
pub struct FetchCommitteesResponse {
    // // Serialized consecutive Commit.
    // #[prost(bytes = "bytes", repeated, tag = "1")]
    // pub commits: Vec<Bytes>,
    // // Serialized SignedBlock that certify the last commit from above.
    // #[prost(bytes = "bytes", repeated, tag = "2")]
    // pub certifier_blocks: Vec<Bytes>,
}
