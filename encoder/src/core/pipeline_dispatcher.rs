use crate::{
    messaging::EncoderInternalNetworkClient,
    pipelines::{
        commit::CommitProcessor, commit_votes::CommitVotesProcessor, input::InputProcessor,
        reveal::RevealProcessor, score_vote::ScoreVoteProcessor,
    },
    types::{commit::Commit, commit_votes::CommitVotes, reveal::Reveal, score_vote::ScoreVote},
};
use async_trait::async_trait;
use evaluation::messaging::EvaluationClient;
use inference::client::InferenceClient;
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use tokio_util::sync::CancellationToken;
use types::error::ShardResult;
use types::multiaddr::Multiaddr;
use types::shard::Input;
use types::{
    actors::ActorHandle,
    shard::Shard,
    shard_crypto::{keys::PeerPublicKey, verified::Verified},
};

#[async_trait]
pub trait InternalDispatcher: Sync + Send + 'static {
    async fn dispatch_commit(
        &self,
        shard: Shard,
        commit: Verified<Commit>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
    async fn dispatch_commit_votes(
        &self,
        shard: Shard,
        votes: Verified<CommitVotes>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
    async fn dispatch_reveal(
        &self,
        shard: Shard,
        reveal: Verified<Reveal>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
    async fn dispatch_score_vote(
        &self,
        shard: Shard,
        scores: Verified<ScoreVote>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct InternalPipelineDispatcher<
    E: EncoderInternalNetworkClient,
    O: ObjectNetworkClient,
    S: ObjectStorage,
    P: EvaluationClient,
> {
    commit_handle: ActorHandle<CommitProcessor<O, E, S, P>>,
    commit_votes_handle: ActorHandle<CommitVotesProcessor<O, E, S, P>>,
    reveal_handle: ActorHandle<RevealProcessor<O, E, S, P>>,
    score_vote_handle: ActorHandle<ScoreVoteProcessor<E>>,
}

impl<
        E: EncoderInternalNetworkClient,
        O: ObjectNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > InternalPipelineDispatcher<E, O, S, P>
{
    pub(crate) fn new(
        commit_handle: ActorHandle<CommitProcessor<O, E, S, P>>,
        commit_votes_handle: ActorHandle<CommitVotesProcessor<O, E, S, P>>,
        reveal_handle: ActorHandle<RevealProcessor<O, E, S, P>>,
        score_vote_handle: ActorHandle<ScoreVoteProcessor<E>>,
    ) -> Self {
        Self {
            commit_handle,
            commit_votes_handle,
            reveal_handle,
            score_vote_handle,
        }
    }
}

#[async_trait]
impl<
        E: EncoderInternalNetworkClient,
        C: ObjectNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > InternalDispatcher for InternalPipelineDispatcher<E, C, S, P>
{
    async fn dispatch_commit(
        &self,
        shard: Shard,
        commit: Verified<Commit>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.commit_handle
            .background_process((shard, commit), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_commit_votes(
        &self,
        shard: Shard,
        votes: Verified<CommitVotes>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.commit_votes_handle
            .background_process((shard, votes), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_reveal(
        &self,
        shard: Shard,
        reveal: Verified<Reveal>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.reveal_handle
            .background_process((shard, reveal), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_score_vote(
        &self,
        shard: Shard,
        score_vote: Verified<ScoreVote>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.score_vote_handle
            .background_process((shard, score_vote), cancellation)
            .await?;
        Ok(())
    }
}

#[async_trait]
pub trait ExternalDispatcher: Sync + Send + 'static {
    async fn dispatch_input(
        &self,
        shard: Shard,
        input: Verified<Input>,
        peer: PeerPublicKey,
        address: Multiaddr,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct ExternalPipelineDispatcher<
    E: EncoderInternalNetworkClient,
    O: ObjectNetworkClient,
    M: InferenceClient,
    S: ObjectStorage,
    P: EvaluationClient,
> {
    input_handle: ActorHandle<InputProcessor<E, O, M, S, P>>,
}

impl<
        E: EncoderInternalNetworkClient,
        O: ObjectNetworkClient,
        M: InferenceClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > ExternalPipelineDispatcher<E, O, M, S, P>
{
    pub(crate) fn new(input_handle: ActorHandle<InputProcessor<E, O, M, S, P>>) -> Self {
        Self { input_handle }
    }
}

#[async_trait]
impl<
        E: EncoderInternalNetworkClient,
        O: ObjectNetworkClient,
        M: InferenceClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > ExternalDispatcher for ExternalPipelineDispatcher<E, O, M, S, P>
{
    async fn dispatch_input(
        &self,
        shard: Shard,
        input: Verified<Input>,
        peer: PeerPublicKey,
        address: Multiaddr,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.input_handle
            .background_process((shard, input, peer, address), cancellation)
            .await?;
        Ok(())
    }
}
