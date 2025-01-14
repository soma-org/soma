use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::{
    actors::{pipelines::shard_input::ShardInputProcessor, ActorHandle},
    error::ShardResult,
    intelligence::model::Model,
    networking::{blob::ObjectNetworkClient, messaging::EncoderNetworkClient},
    storage::object::ObjectStorage,
    types::{
        network_committee::NetworkingIndex, shard::Shard, shard_input::ShardInput, signed::Signed,
        verified::Verified,
    },
};

#[async_trait]
pub trait PipelineDispatcher: Sync + Send + 'static {
    async fn shard_input(
        &self,
        networking_index: NetworkingIndex,
        shard: Shard,
        shard_input: Verified<Signed<ShardInput>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()>;
}

#[derive(Clone)]
pub(crate) struct ActorPipelineDispatcher<
    SNC: EncoderNetworkClient,
    M: Model,
    OS: ObjectStorage,
    ONC: ObjectNetworkClient,
> {
    shard_input_handle: ActorHandle<ShardInputProcessor<SNC, M, OS, ONC>>,
}

impl<SNC: EncoderNetworkClient, M: Model, OS: ObjectStorage, ONC: ObjectNetworkClient>
    ActorPipelineDispatcher<SNC, M, OS, ONC>
{
    pub(crate) fn new(
        shard_input_handle: ActorHandle<ShardInputProcessor<SNC, M, OS, ONC>>,
    ) -> Self {
        Self { shard_input_handle }
    }
}

#[async_trait]
impl<SNC: EncoderNetworkClient, M: Model, OS: ObjectStorage, ONC: ObjectNetworkClient>
    PipelineDispatcher for ActorPipelineDispatcher<SNC, M, OS, ONC>
{
    async fn shard_input(
        &self,
        networking_index: NetworkingIndex,
        shard: Shard,
        shard_input: Verified<Signed<ShardInput>>,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        let input = (networking_index, shard, shard_input);
        self.shard_input_handle
            .background_process(input, cancellation)
            .await;
        Ok(())
    }
}
