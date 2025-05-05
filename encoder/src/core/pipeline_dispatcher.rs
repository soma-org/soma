use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use model::client::ModelClient;
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use probe::messaging::ProbeClient;
use shared::{
    crypto::keys::PeerPublicKey, probe::ProbeMetadata, signed::Signed, verified::Verified,
};
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
        probe_metadata: ProbeMetadata,
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
pub(crate) struct InternalPipelineDispatcher<
    E: EncoderInternalNetworkClient,
    C: ObjectNetworkClient,
    S: ObjectStorage,
    P: ProbeClient,
> {
    certified_commit_handle: ActorHandle<CommitProcessor<E, C, S, P>>,
    commit_votes_handle: ActorHandle<CommitVotesProcessor<E, S, P>>,
    reveal_handle: ActorHandle<RevealProcessor<E, S, P>>,
    reveal_votes_handle: ActorHandle<RevealVotesProcessor<E, S, P>>,
    scores_handle: ActorHandle<ScoresProcessor<E>>,
}

impl<E: EncoderInternalNetworkClient, C: ObjectNetworkClient, S: ObjectStorage, P: ProbeClient>
    InternalPipelineDispatcher<E, C, S, P>
{
    pub(crate) fn new(
        certified_commit_handle: ActorHandle<CommitProcessor<E, C, S, P>>,
        commit_votes_handle: ActorHandle<CommitVotesProcessor<E, S, P>>,
        reveal_handle: ActorHandle<RevealProcessor<E, S, P>>,
        reveal_votes_handle: ActorHandle<RevealVotesProcessor<E, S, P>>,
        scores_handle: ActorHandle<ScoresProcessor<E>>,
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
impl<E: EncoderInternalNetworkClient, C: ObjectNetworkClient, S: ObjectStorage, P: ProbeClient>
    InternalDispatcher for InternalPipelineDispatcher<E, C, S, P>
{
    async fn dispatch_commit(
        &self,
        shard: Shard,
        commit: Verified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
        probe_metadata: ProbeMetadata,
        peer: PeerPublicKey,
        address: Multiaddr,
    ) -> ShardResult<()> {
        let cancellation = CancellationToken::new();
        self.certified_commit_handle
            .background_process((shard, commit, probe_metadata, peer, address), cancellation)
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
        probe_metadata: ProbeMetadata,
        peer: PeerPublicKey,
        address: Multiaddr,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct ExternalPipelineDispatcher<
    E: EncoderInternalNetworkClient,
    O: ObjectNetworkClient,
    M: ModelClient,
    S: ObjectStorage,
    P: ProbeClient,
> {
    input_handle: ActorHandle<InputProcessor<E, O, M, S, P>>,
}

impl<
        E: EncoderInternalNetworkClient,
        O: ObjectNetworkClient,
        M: ModelClient,
        S: ObjectStorage,
        P: ProbeClient,
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
        M: ModelClient,
        S: ObjectStorage,
        P: ProbeClient,
    > ExternalDispatcher for ExternalPipelineDispatcher<E, O, M, S, P>
{
    async fn dispatch_input(
        &self,
        shard: Shard,
        input: Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
        probe_metadata: ProbeMetadata,
        peer: PeerPublicKey,
        address: Multiaddr,
    ) -> ShardResult<()> {
        let cancellation = CancellationToken::new();
        self.input_handle
            .background_process((shard, input, probe_metadata, peer, address), cancellation)
            .await?;
        Ok(())
    }
}
