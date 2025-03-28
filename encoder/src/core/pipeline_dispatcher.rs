use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{
    crypto::keys::{EncoderPublicKey, PeerPublicKey},
    metadata::Metadata,
    probe::ProbeMetadata,
    signed::Signed,
    verified::Verified,
};
use tokio_util::sync::CancellationToken;

use crate::{
    actors::{
        pipelines::{
            certified_commit::CertifiedCommitProcessor, commit_votes::CommitVotesProcessor,
            input::InputProcessor, reveal::RevealProcessor, reveal_votes::RevealVotesProcessor,
            scores::ScoresProcessor,
        },
        ActorHandle,
    },
    error::ShardResult,
    intelligence::model::Model,
    messaging::EncoderInternalNetworkClient,
    networking::object::ObjectNetworkClient,
    storage::object::ObjectStorage,
    types::{
        certified::Certified,
        shard::Shard,
        shard_commit::ShardCommit,
        shard_input::ShardInput,
        shard_reveal::ShardReveal,
        shard_scores::ShardScores,
        shard_verifier::ShardAuthToken,
        shard_votes::{CommitRound, RevealRound, ShardVotes},
    },
};

#[async_trait]
pub trait InternalDispatcher: Sync + Send + 'static {
    async fn dispatch_certified_commit(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        probe_metadata: ProbeMetadata,
        certified_commit: Verified<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>,
    ) -> ShardResult<()>;
    async fn dispatch_commit_votes(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        votes: Verified<Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    async fn dispatch_reveal(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        metadata: Metadata,
        reveal: Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    async fn dispatch_reveal_votes(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        votes: Verified<Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    async fn dispatch_scores(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        scores: Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct InternalPipelineDispatcher<
    E: EncoderInternalNetworkClient,
    O: ObjectNetworkClient,
    S: ObjectStorage,
> {
    certified_commit_handle: ActorHandle<CertifiedCommitProcessor<E, O, S>>,
    commit_votes_handle: ActorHandle<CommitVotesProcessor>,
    reveal_handle: ActorHandle<RevealProcessor>,
    reveal_votes_handle: ActorHandle<RevealVotesProcessor<E, S>>,
    scores_handle: ActorHandle<ScoresProcessor>,
}

impl<E: EncoderInternalNetworkClient, O: ObjectNetworkClient, S: ObjectStorage>
    InternalPipelineDispatcher<E, O, S>
{
    pub(crate) fn new(
        certified_commit_handle: ActorHandle<CertifiedCommitProcessor<E, O, S>>,
        commit_votes_handle: ActorHandle<CommitVotesProcessor>,
        reveal_handle: ActorHandle<RevealProcessor>,
        reveal_votes_handle: ActorHandle<RevealVotesProcessor<E, S>>,
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
impl<E: EncoderInternalNetworkClient, O: ObjectNetworkClient, S: ObjectStorage> InternalDispatcher
    for InternalPipelineDispatcher<E, O, S>
{
    async fn dispatch_certified_commit(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        probe_metadata: ProbeMetadata,
        certified_commit: Verified<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
        let cancellation = CancellationToken::new();
        self.certified_commit_handle
            .background_process(
                (auth_token, shard, probe_metadata, certified_commit),
                cancellation,
            )
            .await?;
        Ok(())
    }
    async fn dispatch_commit_votes(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        votes: Verified<Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
        let cancellation = CancellationToken::new();
        self.commit_votes_handle
            .background_process((auth_token, shard, votes), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_reveal(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        metadata: Metadata,
        reveal: Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
        let cancellation = CancellationToken::new();
        self.reveal_handle
            .background_process((auth_token, shard, metadata, reveal), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_reveal_votes(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        votes: Verified<Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
        let cancellation = CancellationToken::new();
        self.reveal_votes_handle
            .background_process((auth_token, shard, votes), cancellation)
            .await?;
        Ok(())
    }
    async fn dispatch_scores(
        &self,
        encoder: &EncoderPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        scores: Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
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
        peer: &PeerPublicKey,
        auth_token: ShardAuthToken,
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
        peer: &PeerPublicKey,
        auth_token: ShardAuthToken,
        shard: Shard,
        input: Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()> {
        // TODO: use or remove peer
        // TODO: need to create correct child cancellation token here
        let cancellation = CancellationToken::new();
        self.input_handle
            .background_process((auth_token, shard, input), cancellation)
            .await?;
        Ok(())
    }
}
