use async_trait::async_trait;
use fastcrypto::{bls12381::min_sig, ed25519::Ed25519Signature};
use shared::{network_committee::NetworkingIndex, signed::Signed, verified::Verified};
use tokio_util::sync::CancellationToken;

use crate::{
    actors::{pipelines::shard_input::ShardInputProcessor, ActorHandle},
    error::ShardResult,
    intelligence::model::Model,
    networking::{messaging::EncoderInternalNetworkClient, object::ObjectNetworkClient},
    storage::object::ObjectStorage,
    types::{shard::Shard, shard_input::ShardInput},
};

#[async_trait]
pub trait PipelineDispatcher: Sync + Send + 'static {
    async fn shard_input(
        &self,
        networking_index: NetworkingIndex,
        shard: Shard,
        shard_input: Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct ActorPipelineDispatcher<
    SNC: EncoderInternalNetworkClient,
    M: Model,
    OS: ObjectStorage,
    ONC: ObjectNetworkClient,
> {
    shard_input_handle: ActorHandle<ShardInputProcessor<SNC, M, OS, ONC>>,
}

impl<SNC: EncoderInternalNetworkClient, M: Model, OS: ObjectStorage, ONC: ObjectNetworkClient>
    ActorPipelineDispatcher<SNC, M, OS, ONC>
{
    pub(crate) fn new(
        shard_input_handle: ActorHandle<ShardInputProcessor<SNC, M, OS, ONC>>,
    ) -> Self {
        Self { shard_input_handle }
    }
}

#[async_trait]
impl<SNC: EncoderInternalNetworkClient, M: Model, OS: ObjectStorage, ONC: ObjectNetworkClient>
    PipelineDispatcher for ActorPipelineDispatcher<SNC, M, OS, ONC>
{
    async fn shard_input(
        &self,
        networking_index: NetworkingIndex,
        shard: Shard,
        shard_input: Verified<Signed<ShardInput, min_sig::BLS12381Signature>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        let input = (networking_index, shard, shard_input);
        self.shard_input_handle
            .background_process(input, cancellation)
            .await;
        Ok(())
    }
}
