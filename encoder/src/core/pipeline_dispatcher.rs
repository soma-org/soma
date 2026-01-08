use crate::{
    messaging::EncoderInternalNetworkClient,
    pipelines::{
        commit::CommitProcessor, commit_votes::CommitVotesProcessor, input::InputProcessor,
        report_vote::ReportVoteProcessor, reveal::RevealProcessor,
    },
    types::{commit::Commit, commit_votes::CommitVotes, report_vote::ReportVote, reveal::Reveal},
};
use async_trait::async_trait;
use intelligence::{
    evaluation::networking::EvaluationClient, inference::networking::InferenceClient,
};
use tokio_util::sync::CancellationToken;
use types::error::ShardResult;
use types::shard::Input;
use types::{actors::ActorHandle, shard::Shard, shard_crypto::verified::Verified};

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
        commit_votes: Verified<CommitVotes>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
    async fn dispatch_reveal(
        &self,
        shard: Shard,
        reveal: Verified<Reveal>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
    async fn dispatch_report_vote(
        &self,
        shard: Shard,
        report_vote: Verified<ReportVote>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct InternalPipelineDispatcher<C: EncoderInternalNetworkClient, E: EvaluationClient> {
    commit_handle: ActorHandle<CommitProcessor<C, E>>,
    commit_votes_handle: ActorHandle<CommitVotesProcessor<C, E>>,
    reveal_handle: ActorHandle<RevealProcessor<C, E>>,
    report_vote_handle: ActorHandle<ReportVoteProcessor<C>>,
}

impl<C: EncoderInternalNetworkClient, E: EvaluationClient> InternalPipelineDispatcher<C, E> {
    pub(crate) fn new(
        commit_handle: ActorHandle<CommitProcessor<C, E>>,
        commit_votes_handle: ActorHandle<CommitVotesProcessor<C, E>>,
        reveal_handle: ActorHandle<RevealProcessor<C, E>>,
        report_vote_handle: ActorHandle<ReportVoteProcessor<C>>,
    ) -> Self {
        Self {
            commit_handle,
            commit_votes_handle,
            reveal_handle,
            report_vote_handle,
        }
    }
}

#[async_trait]
impl<C: EncoderInternalNetworkClient, E: EvaluationClient> InternalDispatcher
    for InternalPipelineDispatcher<C, E>
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
    async fn dispatch_report_vote(
        &self,
        shard: Shard,
        report_vote: Verified<ReportVote>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.report_vote_handle
            .background_process((shard, report_vote), cancellation)
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
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct ExternalPipelineDispatcher<
    C: EncoderInternalNetworkClient,
    E: EvaluationClient,
    I: InferenceClient,
> {
    input_handle: ActorHandle<InputProcessor<C, E, I>>,
}

impl<C: EncoderInternalNetworkClient, E: EvaluationClient, I: InferenceClient>
    ExternalPipelineDispatcher<C, E, I>
{
    pub(crate) fn new(input_handle: ActorHandle<InputProcessor<C, E, I>>) -> Self {
        Self { input_handle }
    }
}

#[async_trait]
impl<C: EncoderInternalNetworkClient, E: EvaluationClient, I: InferenceClient> ExternalDispatcher
    for ExternalPipelineDispatcher<C, E, I>
{
    async fn dispatch_input(
        &self,
        shard: Shard,
        input: Verified<Input>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.input_handle
            .background_process((shard, input), cancellation)
            .await?;
        Ok(())
    }
}
