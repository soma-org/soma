use crate::{datastore::Store, types::finality::Finality};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use quick_cache::sync::{Cache, GuardResult};
use shared::{
    actors::{ActorMessage, Processor},
    digest::Digest,
    error::{ShardError, ShardResult},
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use std::{sync::Arc, time::Duration};
use tracing::info;

pub(crate) struct FinalityProcessor {
    store: Arc<dyn Store>,
    recv_dedup: Cache<Digest<Shard>, ()>,
}
impl FinalityProcessor {
    pub(crate) fn new(store: Arc<dyn Store>, recv_cache_capacity: usize) -> Self {
        Self {
            store,
            recv_dedup: Cache::new(recv_cache_capacity),
        }
    }
}

#[async_trait]
impl Processor for FinalityProcessor {
    type Input = (
        Shard,
        Verified<Signed<Finality, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, _verified_signed_finality) = msg.input;
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
            // info!("Performing mock clean up for shard: {}", shard_digest);
            // TODO: msg.cancellation.cancel();
            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
