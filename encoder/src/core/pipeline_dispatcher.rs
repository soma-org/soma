use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use shared::{crypto::keys::PeerPublicKey, signed::Signed, verified::Verified};
use soma_network::multiaddr::Multiaddr;
use tokio_util::sync::CancellationToken;

use crate::{
    actors::{
        pipelines::{
            commit::CommitProcessor, commit_votes::CommitVotesProcessor, input::InputProcessor,
            reveal::RevealProcessor, reveal_votes::RevealVotesProcessor, scores::ScoresProcessor,
        },
        ActorHandle,
    },
    error::ShardResult,
    intelligence::model::Model,
    messaging::EncoderInternalNetworkClient,
    types::{
        shard::Shard, shard_commit::ShardCommit, shard_commit_votes::ShardCommitVotes,
        shard_input::ShardInput, shard_reveal::ShardReveal, shard_reveal_votes::ShardRevealVotes,
        shard_scores::ShardScores,
    },
};

#[async_trait]
pub trait InternalDispatcher: Sync + Send + 'static {
    async fn dispatch_commit(
        &self,
        shard: Shard,
        commit: Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        peer: PeerPublicKey,
        address: Multiaddr,
    ) -> ShardResult<()>;
    async fn dispatch_commit_votes(
        &self,
        shard: Shard,
        votes: Verified<Signed<ShardCommitVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    async fn dispatch_reveal(
        &self,
        shard: Shard,
        reveal: Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    async fn dispatch_reveal_votes(
        &self,
        shard: Shard,
        votes: Verified<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    async fn dispatch_scores(
        &self,
        shard: Shard,
        scores: Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct InternalPipelineDispatcher<C: ObjectNetworkClient, S: ObjectStorage> {
    certified_commit_handle: ActorHandle<CommitProcessor<C, S>>,
    commit_votes_handle: ActorHandle<CommitVotesProcessor>,
    reveal_handle: ActorHandle<RevealProcessor>,
    reveal_votes_handle: ActorHandle<RevealVotesProcessor>,
    scores_handle: ActorHandle<ScoresProcessor>,
}

impl<C: ObjectNetworkClient, S: ObjectStorage> InternalPipelineDispatcher<C, S> {
    pub(crate) fn new(
        certified_commit_handle: ActorHandle<CommitProcessor<C, S>>,
        commit_votes_handle: ActorHandle<CommitVotesProcessor>,
        reveal_handle: ActorHandle<RevealProcessor>,
        reveal_votes_handle: ActorHandle<RevealVotesProcessor>,
        scores_handle: ActorHandle<ScoresProcessor>,
    ) -> Self {
        Self {
            certified_commit_handle,
            commit_votes_handle,
            reveal_handle,
            reveal_votes_handle,
            scores_handle,
        }
    }
}

#[async_trait]
impl<C: ObjectNetworkClient, S: ObjectStorage> InternalDispatcher
    for InternalPipelineDispatcher<C, S>
{
    async fn dispatch_commit(
        &self,
        shard: Shard,
        commit: Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        peer: PeerPublicKey,
        address: Multiaddr,
    ) -> ShardResult<()> {
        let cancellation = CancellationToken::new();
        self.certified_commit_handle
            .background_process((shard, commit, peer, address), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_commit_votes(
        &self,
        shard: Shard,
        votes: Verified<Signed<ShardCommitVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let cancellation = CancellationToken::new();
        self.commit_votes_handle
            .background_process((shard, votes), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_reveal(
        &self,
        shard: Shard,
        reveal: Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let cancellation = CancellationToken::new();
        self.reveal_handle
            .background_process((shard, reveal), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_reveal_votes(
        &self,
        shard: Shard,
        votes: Verified<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let cancellation = CancellationToken::new();
        self.reveal_votes_handle
            .background_process((shard, votes), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_scores(
        &self,
        shard: Shard,
        scores: Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let cancellation = CancellationToken::new();
        self.scores_handle
            .background_process((shard, scores), cancellation)
            .await?;
        Ok(())
    }
}

#[async_trait]
pub trait ExternalDispatcher: Sync + Send + 'static {
    async fn dispatch_input(
        &self,
        shard: Shard,
        input: Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct ExternalPipelineDispatcher<
    O: ObjectNetworkClient,
    M: Model,
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
> {
    input_handle: ActorHandle<InputProcessor<O, M, E, S>>,
}

impl<O: ObjectNetworkClient, M: Model, E: EncoderInternalNetworkClient, S: ObjectStorage>
    ExternalPipelineDispatcher<O, M, E, S>
{
    pub(crate) fn new(input_handle: ActorHandle<InputProcessor<O, M, E, S>>) -> Self {
        Self { input_handle }
    }
}

#[async_trait]
impl<O: ObjectNetworkClient, M: Model, E: EncoderInternalNetworkClient, S: ObjectStorage>
    ExternalDispatcher for ExternalPipelineDispatcher<O, M, E, S>
{
    async fn dispatch_input(
        &self,
        shard: Shard,
        input: Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        let cancellation = CancellationToken::new();
        self.input_handle
            .background_process((shard, input), cancellation)
            .await?;
        Ok(())
    }
}
