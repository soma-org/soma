use bytes::Bytes;

use serde::{Deserialize, Serialize};

use crate::{
    accumulator::CommitIndex,
    committee::EpochId,
    consensus::{block::Round, commit::CommitDigest},
};

pub type CommitTimestamp = u64;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GetCommitSummaryRequest {
    Latest,
    ByDigest(CommitDigest),
    ByIndex(CommitIndex),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCommitAvailabilityResponse {
    pub highest_synced_commit: CommitIndex,
    pub lowest_available_commit: CommitIndex,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PushCommitSummaryResponse {
    pub timestamp_ms: CommitTimestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCommitAvailabilityRequest {
    pub timestamp_ms: CommitTimestamp,
}

#[derive(Clone, prost::Message, Serialize, Deserialize)]
pub struct FetchBlocksRequest {
    #[prost(bytes = "vec", repeated, tag = "1")]
    pub block_refs: Vec<Vec<u8>>,
    // The highest accepted round per authority. The vector represents the round for each authority
    // and its length should be the same as the committee size.
    #[prost(uint32, repeated, tag = "2")]
    pub highest_accepted_rounds: Vec<Round>,

    #[prost(uint64, tag = "3")]
    pub epoch: EpochId,
}

#[derive(Clone, prost::Message, Serialize, Deserialize)]
pub struct FetchBlocksResponse {
    // The response of the requested blocks as Serialized SignedBlock.
    #[prost(bytes = "bytes", repeated, tag = "1")]
    pub blocks: Vec<Bytes>,
}

#[derive(Clone, prost::Message, Serialize, Deserialize)]
pub struct FetchCommitsRequest {
    #[prost(uint32, tag = "1")]
    pub start: CommitIndex,
    #[prost(uint32, tag = "2")]
    pub end: CommitIndex,
}

#[derive(Clone, prost::Message, Serialize, Deserialize)]
pub struct FetchCommitsResponse {
    // Serialized consecutive Commit.
    #[prost(bytes = "bytes", repeated, tag = "1")]
    pub commits: Vec<Bytes>,
    // Serialized SignedBlock that certify the last commit from above.
    #[prost(bytes = "bytes", repeated, tag = "2")]
    pub certifier_blocks: Vec<Bytes>,
}
