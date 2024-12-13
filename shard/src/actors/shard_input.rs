use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::{core::encoder_core::EncoderCore, intelligence::model::Model, networking::messaging::EncoderNetworkClient, storage::blob::BlobStorage, types::{shard::Shard, shard_input::ShardInput, signed::Signed, verified::Verified}};

use super::{ActorMessage, Processor};

use async_trait::async_trait;



pub(crate) struct ShardInputProcessor<C: EncoderNetworkClient, M: Model, B: BlobStorage> {
    core: EncoderCore<C, M, B>,
    semaphore: Arc<Semaphore>,
}

impl<C: EncoderNetworkClient, M: Model, B: BlobStorage> ShardInputProcessor<C, M, B> {
    pub fn new(
        core: EncoderCore<C, M, B>,
        concurrency: usize,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(concurrency));
        Self {
            core,
            semaphore,
        }
    }
}

#[async_trait]
impl<C: EncoderNetworkClient, M: Model, B: BlobStorage> Processor for ShardInputProcessor<C, M, B> {
    type Input = (Shard, Verified<Signed<ShardInput>>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        if let Ok(permit) = self.semaphore.clone().acquire_owned().await {
            let core = self.core.clone();
            tokio::spawn(async move {
                let result = core.process_shard_input(msg.input.0, msg.input.1).await;
                let _ = msg.sender.send(result);
                drop(permit);
            });
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
