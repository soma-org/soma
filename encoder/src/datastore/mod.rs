#![doc = include_str!("README.md")]

pub(crate) mod mem_store;
pub(crate) mod rocksdb_store;

use std::collections::HashMap;
use std::time::Instant;

use crate::types::commit_votes::CommitVotes;
use crate::types::report_vote::ReportVote;
use serde::{Deserialize, Serialize};
use types::error::ShardResult;
use types::metadata::DownloadMetadata;
use types::shard::Input;
use types::submission::Submission;
use types::{
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        keys::{EncoderAggregateSignature, EncoderPublicKey},
        verified::Verified,
    },
};

#[derive(Ord, PartialOrd, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub(crate) enum ShardStage {
    Input,
    Commit,
    CommitVote,
    Reveal,
    Evaluation,
    ReportVote,
    Finalize,
}

/// The store is a common interface for accessing encoder data
pub trait Store: Send + Sync + 'static {
    fn add_shard_stage_message(
        &self,
        shard: &Shard,
        stage: ShardStage,
        encoder: &EncoderPublicKey,
    ) -> ShardResult<()>;

    fn add_shard_stage_dispatch(&self, shard: &Shard, stage: ShardStage) -> ShardResult<()>;

    fn add_input(&self, shard: &Shard, input: Verified<Input>) -> ShardResult<()>;

    fn get_input(&self, shard: &Shard) -> ShardResult<Verified<Input>>;

    fn add_submission_digest(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
        submission_digest: Digest<Submission>,
    ) -> ShardResult<()>;

    fn get_submission_digest(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
    ) -> ShardResult<(Digest<Submission>, Instant)>;

    fn get_all_submission_digests(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<(EncoderPublicKey, Digest<Submission>, Instant)>>;

    fn add_commit_votes(&self, shard: &Shard, votes: &Verified<CommitVotes>) -> ShardResult<()>;

    fn get_all_commit_votes(&self, shard: &Shard) -> ShardResult<Vec<CommitVotes>>;

    fn add_accepted_commit(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
        submission_digest: Digest<Submission>,
    ) -> ShardResult<()>;

    fn get_all_accepted_commits(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<(EncoderPublicKey, Digest<Submission>)>>;

    fn add_submission(&self, shard: &Shard, submission: Submission) -> ShardResult<()>;

    fn get_submission(
        &self,
        shard: &Shard,
        submission_digest: Digest<Submission>,
    ) -> ShardResult<(Submission, Instant)>;

    fn get_all_submissions(&self, shard: &Shard) -> ShardResult<Vec<(Submission, Instant)>>;

    fn add_report_vote(&self, shard: &Shard, report_vote: &Verified<ReportVote>)
        -> ShardResult<()>;
    fn get_all_report_votes(&self, shard: &Shard) -> ShardResult<Vec<ReportVote>>;

    fn add_aggregate_score(
        &self,
        shard: &Shard,
        agg_details: (EncoderAggregateSignature, Vec<EncoderPublicKey>),
    ) -> ShardResult<()>;
    fn get_aggregate_score(
        &self,
        shard: &Shard,
    ) -> ShardResult<(EncoderAggregateSignature, Vec<EncoderPublicKey>)>;
}
