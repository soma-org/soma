use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{signed::Signed, verified::Verified};
use tokio_util::sync::CancellationToken;

use crate::{
    actors::{
        pipelines::{
            certified_commit::CertifiedCommitProcessor, commit_votes::CommitVotesProcessor,
            reveal::RevealProcessor, reveal_votes::RevealVotesProcessor,
        },
        ActorHandle,
    },
    error::ShardResult,
    types::{
        certified::Certified,
        encoder_committee::EncoderIndex,
        shard_commit::ShardCommit,
        shard_reveal::ShardReveal,
        shard_votes::{CommitRound, RevealRound, ShardVotes},
    },
};

#[async_trait]
pub trait Dispatcher: Sync + Send + 'static {
    async fn dispatch_certified_commit(
        &self,
        peer: EncoderIndex,
        certified_commit: Verified<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>,
    ) -> ShardResult<()>;
    async fn dispatch_commit_votes(
        &self,
        peer: EncoderIndex,
        votes: Verified<Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    async fn dispatch_reveal(
        &self,
        peer: EncoderIndex,
        reveal: Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    async fn dispatch_reveal_votes(
        &self,
        peer: EncoderIndex,
        votes: Verified<Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct PipelineDispatcher {
    certified_commit_handle: ActorHandle<CertifiedCommitProcessor>,
    commit_votes_handle: ActorHandle<CommitVotesProcessor>,
    reveal_handle: ActorHandle<RevealProcessor>,
    reveal_votes_handle: ActorHandle<RevealVotesProcessor>,
}

impl PipelineDispatcher {
    pub(crate) fn new(
        certified_commit_handle: ActorHandle<CertifiedCommitProcessor>,
        commit_votes_handle: ActorHandle<CommitVotesProcessor>,
        reveal_handle: ActorHandle<RevealProcessor>,
        reveal_votes_handle: ActorHandle<RevealVotesProcessor>,
    ) -> Self {
        Self {
            certified_commit_handle,
            commit_votes_handle,
            reveal_handle,
            reveal_votes_handle,
        }
    }
}

#[async_trait]
impl Dispatcher for PipelineDispatcher {
    async fn dispatch_certified_commit(
        &self,
        peer: EncoderIndex,
        certified_commit: Verified<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
        let cancellation = CancellationToken::new();
        self.certified_commit_handle
            .background_process(certified_commit, cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_commit_votes(
        &self,
        peer: EncoderIndex,
        votes: Verified<Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
        let cancellation = CancellationToken::new();
        self.commit_votes_handle
            .background_process(votes, cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_reveal(
        &self,
        peer: EncoderIndex,
        reveal: Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
        let cancellation = CancellationToken::new();
        self.reveal_handle
            .background_process(reveal, cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_reveal_votes(
        &self,
        peer: EncoderIndex,
        votes: Verified<Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
        let cancellation = CancellationToken::new();
        self.reveal_votes_handle
            .background_process(votes, cancellation)
            .await?;
        Ok(())
    }
}
