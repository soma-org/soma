use crate::datastore::Store;
use async_trait::async_trait;
use quick_cache::sync::{Cache, GuardResult};
use std::{sync::Arc, time::Duration};
use tracing::info;
use types::{
    actors::{ActorMessage, Processor},
    error::{ShardError, ShardResult},
    shard::Shard,
    shard_crypto::digest::Digest,
};

pub(crate) struct CleanUpProcessor {
    store: Arc<dyn Store>,
    recv_dedup: Cache<Digest<Shard>, ()>,
}
impl CleanUpProcessor {
    pub(crate) fn new(store: Arc<dyn Store>, recv_cache_capacity: usize) -> Self {
        Self {
            store,
            recv_dedup: Cache::new(recv_cache_capacity),
        }
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
            match self
                .recv_dedup
                .get_value_or_guard(&shard_digest, Some(Duration::from_secs(5)))
            {
                GuardResult::Value(_) => return Err(ShardError::RecvDuplicate),
                GuardResult::Guard(placeholder) => {
                    placeholder.insert(());
                }
                GuardResult::Timeout => (),
            }
            info!("Performing mock clean up for shard: {}", shard_digest);
            // TODO: msg.cancellation.cancel();
            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
