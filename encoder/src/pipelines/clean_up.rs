use crate::datastore::Store;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::info;
use types::{
    actors::{ActorMessage, Processor},
    error::ShardResult,
    shard::Shard,
};

pub(crate) struct CleanUpProcessor {
    store: Arc<dyn Store>,
}
impl CleanUpProcessor {
    pub(crate) fn new(store: Arc<dyn Store>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Processor for CleanUpProcessor {
    type Input = Shard;
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let shard = msg.input;
            let shard_digest = shard.digest()?;
            info!("Performing mock clean up for shard: {}", shard_digest);
            // TODO: msg.cancellation.cancel();
            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
