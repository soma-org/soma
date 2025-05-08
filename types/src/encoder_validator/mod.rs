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
    #[prost(message, repeated, tag = "1")]
    pub epoch_committees: Vec<EpochCommittee>,
}

#[derive(Clone, prost::Message, Serialize, Deserialize)]
pub struct EpochCommittee {
    #[prost(uint64, tag = "1")]
    pub epoch: EpochId,
    #[prost(bytes, tag = "2")]
    pub validator_set: Bytes, // Serialized ValidatorSet
    #[prost(bytes, tag = "3")]
    pub aggregate_signature: Bytes, // Serialized AggregateAuthoritySignature for validator set
    #[prost(uint64, tag = "4")]
    pub next_epoch_start_timestamp_ms: u64,
    #[prost(uint32, repeated, tag = "5")]
    pub signer_indices: Vec<u32>, // AuthorityIndex values of signers
    #[prost(bytes, tag = "6")]
    pub encoder_committee: Bytes, // Serialized EncoderCommittee
    #[prost(bytes, tag = "7")]
    pub encoder_aggregate_signature: Bytes, // Serialized AggregateAuthoritySignature for encoder committee
}
