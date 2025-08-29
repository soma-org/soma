use crate::{
    messaging::EncoderInternalNetworkClient,
    pipelines::{
        commit::CommitProcessor, commit_votes::CommitVotesProcessor, finality::FinalityProcessor,
        input::InputProcessor, reveal::RevealProcessor, scores::ScoresProcessor,
    },
    types::{
        commit::Commit, commit_votes::CommitVotes, finality::Finality, input::Input, reveal::Reveal,
    },
};
use async_trait::async_trait;
use evaluation::messaging::EvaluationClient;
use fastcrypto::bls12381::min_sig;
use inference::client::InferenceClient;
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use shared::error::ShardResult;
use shared::{
    actors::ActorHandle, crypto::keys::PeerPublicKey, shard::Shard, signed::Signed,
    verified::Verified,
};
use soma_network::multiaddr::Multiaddr;
use tokio_util::sync::CancellationToken;
use types::shard_score::ShardScore;

#[async_trait]
pub trait InternalDispatcher: Sync + Send + 'static {
    async fn dispatch_commit(
        &self,
        shard: Shard,
        commit: Verified<Signed<Commit, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
    async fn dispatch_commit_votes(
        &self,
        shard: Shard,
        votes: Verified<Signed<CommitVotes, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
    async fn dispatch_reveal(
        &self,
        shard: Shard,
        reveal: Verified<Signed<Reveal, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
    async fn dispatch_scores(
        &self,
        shard: Shard,
        scores: Verified<Signed<ShardScore, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
    async fn dispatch_finality(
        &self,
        shard: Shard,
        scores: Verified<Signed<Finality, min_sig::BLS12381Signature>>,
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
    certified_commit_handle: ActorHandle<CommitProcessor<O, E, S, P>>,
    commit_votes_handle: ActorHandle<CommitVotesProcessor<O, E, S, P>>,
    reveal_handle: ActorHandle<RevealProcessor<O, E, S, P>>,
    scores_handle: ActorHandle<ScoresProcessor<E>>,
    finality_handle: ActorHandle<FinalityProcessor>,
}

impl<
        E: EncoderInternalNetworkClient,
        O: ObjectNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > InternalPipelineDispatcher<E, O, S, P>
{
    pub(crate) fn new(
        certified_commit_handle: ActorHandle<CommitProcessor<O, E, S, P>>,
        commit_votes_handle: ActorHandle<CommitVotesProcessor<O, E, S, P>>,
        reveal_handle: ActorHandle<RevealProcessor<O, E, S, P>>,
        scores_handle: ActorHandle<ScoresProcessor<E>>,
        finality_handle: ActorHandle<FinalityProcessor>,
    ) -> Self {
        Self {
            certified_commit_handle,
            commit_votes_handle,
            reveal_handle,
            scores_handle,
            finality_handle,
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
        commit: Verified<Signed<Commit, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.certified_commit_handle
            .background_process((shard, commit), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_commit_votes(
        &self,
        shard: Shard,
        votes: Verified<Signed<CommitVotes, min_sig::BLS12381Signature>>,
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
        reveal: Verified<Signed<Reveal, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.reveal_handle
            .background_process((shard, reveal), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_scores(
        &self,
        shard: Shard,
        scores: Verified<Signed<ShardScore, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.scores_handle
            .background_process((shard, scores), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_finality(
        &self,
        shard: Shard,
        finality: Verified<Signed<Finality, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        self.finality_handle
            .background_process((shard, finality), cancellation)
            .await?;
        Ok(())
    }
}

#[async_trait]
pub trait ExternalDispatcher: Sync + Send + 'static {
    async fn dispatch_input(
        &self,
        shard: Shard,
        input: Verified<Signed<Input, min_sig::BLS12381Signature>>,
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
        input: Verified<Signed<Input, min_sig::BLS12381Signature>>,
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
