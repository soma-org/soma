#![doc = include_str!("README.md")]

pub(crate) mod mem_store;

use std::time::Instant;

use crate::types::score_vote::ScoreVote;
use crate::types::{commit::Commit, commit_votes::CommitVotes, reveal::Reveal};
use fastcrypto::bls12381::min_sig;
use shared::error::ShardResult;
use shared::{
    crypto::keys::{EncoderAggregateSignature, EncoderPublicKey},
    digest::Digest,
    shard::Shard,
    signed::Signed,
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
    fn lock_signed_commit(
        &self,
        shard: &Shard,
        signed_commit: &Signed<Commit, min_sig::BLS12381Signature>,
    ) -> ShardResult<()>;
    /// adds the signed commit, returns an error if there is a preexisting different signed commit
    fn add_signed_commit(
        &self,
        shard: &Shard,
        signed_commit: &Verified<Signed<Commit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;

    fn count_signed_commits(&self, shard: &Shard) -> ShardResult<usize>;

    fn get_signed_commits(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<Commit, min_sig::BLS12381Signature>>>;

    fn add_first_commit_time(&self, shard: &Shard) -> ShardResult<()>;

    fn get_first_commit_time(&self, shard: &Shard) -> ShardResult<Instant>;

    fn add_signed_reveal(
        &self,
        shard: &Shard,
        signed_reveal: &Verified<Signed<Reveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;

    fn count_signed_reveal(&self, shard: &Shard) -> ShardResult<usize>;

    fn get_signed_reveals(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<Reveal, min_sig::BLS12381Signature>>>;

    fn get_encoder_signed_reveal(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
    ) -> ShardResult<Signed<Reveal, min_sig::BLS12381Signature>>;

    fn add_first_reveal_time(&self, shard: &Shard) -> ShardResult<()>;

    fn get_first_reveal_time(&self, shard: &Shard) -> ShardResult<Instant>;

    fn add_commit_votes(
        &self,
        shard: &Shard,
        votes: &Verified<Signed<CommitVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;

    fn get_commit_votes(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<CommitVotes, min_sig::BLS12381Signature>>>;

    fn get_commit_votes_for_encoder(
        &self,
        shard: &Shard,
        encoder: &EncoderPublicKey,
        digest: Option<&Digest<Signed<Reveal, min_sig::BLS12381Signature>>>,
    ) -> ShardResult<CommitVoteCounts>;

    fn add_signed_score_vote(
        &self,
        shard: &Shard,
        score_vote: &Verified<Signed<ScoreVote, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    fn count_commit_votes(&self, shard: &Shard) -> ShardResult<usize>;
    fn get_signed_score_vote(
        &self,
        shard: &Shard,
    ) -> ShardResult<Vec<Signed<ScoreVote, min_sig::BLS12381Signature>>>;

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
