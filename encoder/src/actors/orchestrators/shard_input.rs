use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::{
    core::encoder_core::EncoderCore,
    intelligence::model::Model,
    networking::{blob::ObjectNetworkClient, messaging::EncoderNetworkClient},
    storage::blob::ObjectStorage,
    types::{
        network_committee::NetworkingIndex, shard::Shard, shard_input::ShardInput, signed::Signed,
        verified::Verified,
    },
};

use crate::actors::{ActorMessage, Processor};

use async_trait::async_trait;

pub(crate) struct ShardInputProcessor<
    C: EncoderNetworkClient,
    M: Model,
    B: ObjectStorage,
    BC: ObjectNetworkClient,
> {
    core: EncoderCore<C, M, B, BC>,
    semaphore: Arc<Semaphore>,
}

impl<C: EncoderNetworkClient, M: Model, B: ObjectStorage, BC: ObjectNetworkClient>
    ShardInputProcessor<C, M, B, BC>
{
    pub fn new(core: EncoderCore<C, M, B, BC>, concurrency: usize) -> Self {
        let semaphore = Arc::new(Semaphore::new(concurrency));
        Self { core, semaphore }
    }
}

#[async_trait]
impl<C: EncoderNetworkClient, M: Model, B: ObjectStorage, BC: ObjectNetworkClient> Processor
    for ShardInputProcessor<C, M, B, BC>
{
    type Input = (NetworkingIndex, Shard, Verified<Signed<ShardInput>>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        if let Ok(permit) = self.semaphore.clone().acquire_owned().await {
            let core = self.core.clone();
            tokio::spawn(async move {
                let result = core
                    .process_shard_input(msg.input.0, msg.input.1, msg.input.2)
                    .await;
                let _ = msg.sender.send(result);
                drop(permit);
            });
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
