#![doc = include_str!("README.md")]

pub(crate) mod mem_store;

use fastcrypto::bls12381::min_sig;
use shared::{signed::Signed, verified::Verified};

use crate::{
    error::ShardResult,
    types::{
        shard::Shard, shard_commit::ShardCommit, shard_commit_votes::ShardCommitVotes,
        shard_reveal::ShardReveal, shard_reveal_votes::ShardRevealVotes,
    },
};

/// The store is a common interface for accessing encoder data
pub(crate) trait Store: Send + Sync + 'static {
    /// lock_signed_commit must return an error if a different commit
    /// already exists. Return ok if the same commit digest exists.
    ///
    /// the internal logic tracks both the committer and the encoder.
    /// the same committer cannot commit twice as well as the encoder that is eligible.
    ///
    /// track digest for encoder and committer seperately such that you can check for
    /// double commits from either party
    fn lock_signed_commit(
        &self,
        shard: &Shard,
        signed_commit: &Signed<ShardCommit, min_sig::BLS12381Signature>,
    ) -> ShardResult<()>;

    /// adds the signed commit, returns an error if there is a preexisting different signed commit
    fn add_signed_commit(
        &self,
        shard: &Shard,
        signed_commit: &Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;

    fn check_reveal_key(
        &self,
        shard: &Shard,
        signed_reveal: &Signed<ShardReveal, min_sig::BLS12381Signature>,
    ) -> ShardResult<()>;

    fn add_signed_reveal(
        &self,
        shard: &Shard,
        signed_reveal: &Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;

    fn add_commit_votes(
        &self,
        shard: &Shard,
        votes: &Verified<Signed<ShardCommitVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;

    fn add_reveal_votes(
        &self,
        shard: &Shard,
        votes: &Verified<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    // ///////////////////////////////
    // fn get_filled_certified_commit_slots(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    // ) -> Vec<EncoderIndex>;
    // fn get_certified_commit(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     slot: EncoderIndex,
    // ) -> ShardResult<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>;
    // fn time_since_first_certified_commit(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    // ) -> Option<Duration>;
    // fn atomic_reveal(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     slot: EncoderIndex,
    //     reveal: EncryptionKey,
    //     checksum: Checksum,
    // ) -> ShardResult<usize>;
    // fn get_filled_reveal_slots(&self, epoch: Epoch, shard_ref: Digest<Shard>) -> Vec<EncoderIndex>;
    // fn get_reveal(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     slot: EncoderIndex,
    // ) -> ShardResult<(EncryptionKey, Checksum)>;
    // fn time_since_first_reveal(&self, epoch: Epoch, shard_ref: Digest<Shard>) -> Option<Duration>;
    // fn add_commit_vote(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     shard: Shard,
    //     vote: ShardVotes<CommitRound>,
    // ) -> ShardResult<(usize, usize)>;
    // fn add_reveal_vote(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     shard: Shard,
    //     vote: ShardVotes<RevealRound>,
    // ) -> ShardResult<(usize, usize)>;
    // fn get_accepted_finalized_reveal_slots(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    // ) -> ShardResult<Vec<EncoderIndex>>;
    // fn add_scores(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     evaluator: EncoderIndex,
    //     signed_scores: Signed<ScoreSet, min_sig::BLS12381Signature>,
    // ) -> ShardResult<Vec<(EncoderIndex, Signed<ScoreSet, min_sig::BLS12381Signature>)>>;
    // fn get_commit_encryption(
    //     &self,
    //     epoch: Epoch,
    //     shard_ref: Digest<Shard>,
    //     evaluator: EncoderIndex,
    //     signed_scores: Signed<ScoreSet, min_sig::BLS12381Signature>,
    // ) -> ShardResult<Vec<(EncoderIndex, Signed<ScoreSet, min_sig::BLS12381Signature>)>>;
}
