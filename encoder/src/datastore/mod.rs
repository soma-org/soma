#![doc = include_str!("README.md")]

pub(crate) mod mem_store;

use std::time::Instant;

use crate::types::score_vote::ScoreVote;
use crate::types::{commit::Commit, commit_votes::CommitVotes, reveal::Reveal};
use shared::error::ShardResult;
use shared::{
    crypto::keys::{EncoderAggregateSignature, EncoderPublicKey},
    digest::Digest,
    shard::Shard,
    verified::Verified,
};

pub(crate) struct CommitVoteCounts {
    accepts: Option<usize>,
    rejects: usize,
    highest: usize,
}
impl CommitVoteCounts {
    pub(crate) fn new(accepts: Option<usize>, rejects: usize, highest: usize) -> Self {
        Self {
            accepts,
            rejects,
            highest,
        }
    }

    pub(crate) fn accept_count(&self) -> Option<usize> {
        self.accepts
    }
    pub(crate) fn reject_count(&self) -> usize {
        self.rejects
    }
    pub(crate) fn highest(&self) -> usize {
        self.highest
    }
}

/// The store is a common interface for accessing encoder data
pub(crate) trait Store: Send + Sync + 'static {
    fn lock_commit(&self, shard: &Shard, commit: &Commit) -> ShardResult<()>;
    /// adds the signed commit, returns an error if there is a preexisting different signed commit
    fn add_commit(&self, shard: &Shard, commit: &Verified<Commit>) -> ShardResult<()>;

    fn count_commits(&self, shard: &Shard) -> ShardResult<usize>;

    fn get_commits(&self, shard: &Shard) -> ShardResult<Vec<Commit>>;

    fn add_first_commit_time(&self, shard: &Shard) -> ShardResult<()>;

    fn get_first_commit_time(&self, shard: &Shard) -> ShardResult<Instant>;

    fn add_reveal(&self, shard: &Shard, reveal: &Verified<Reveal>) -> ShardResult<()>;

    fn count_reveal(&self, shard: &Shard) -> ShardResult<usize>;

    fn get_reveals(&self, shard: &Shard) -> ShardResult<Vec<Reveal>>;

    fn get_encoder_reveal(&self, shard: &Shard, encoder: &EncoderPublicKey) -> ShardResult<Reveal>;

    fn add_first_reveal_time(&self, shard: &Shard) -> ShardResult<()>;

    fn get_first_reveal_time(&self, shard: &Shard) -> ShardResult<Instant>;

    fn add_commit_votes(&self, shard: &Shard, votes: &Verified<CommitVotes>) -> ShardResult<()>;

    fn get_commit_votes(&self, shard: &Shard) -> ShardResult<Vec<CommitVotes>>;

    fn get_commit_votes_for_encoder(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
        digest: Option<&Digest<Reveal>>,
    ) -> ShardResult<CommitVoteCounts>;

    fn add_score_vote(&self, shard: &Shard, score_vote: &Verified<ScoreVote>) -> ShardResult<()>;
    fn count_commit_votes(&self, shard: &Shard) -> ShardResult<usize>;
    fn get_score_vote(&self, shard: &Shard) -> ShardResult<Vec<ScoreVote>>;

    fn add_aggregate_score(
        &self,
        shard: &Shard,
        agg_details: (EncoderAggregateSignature, Vec<EncoderPublicKey>),
    ) -> ShardResult<()>;
    fn get_agg_score(
        &self,
        shard: &Shard,
    ) -> ShardResult<(EncoderAggregateSignature, Vec<EncoderPublicKey>)>;
}
