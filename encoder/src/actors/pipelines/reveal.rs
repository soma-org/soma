use crate::{
    actors::{ActorMessage, Processor},
    core::shard_tracker::ShardTracker,
    datastore::Store,
    error::ShardResult,
    types::{shard::Shard, shard_reveal::ShardReveal},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{signed::Signed, verified::Verified};
use std::sync::Arc;

pub(crate) struct RevealProcessor {
    store: Arc<dyn Store>,
    shard_tracker: ShardTracker,
}

impl RevealProcessor {
    pub(crate) fn new(store: Arc<dyn Store>, shard_tracker: ShardTracker) -> Self {
        Self {
            store,
            shard_tracker,
        }
    }
}

#[async_trait]
impl Processor for RevealProcessor {
    type Input = (
        Shard,
        Verified<Signed<ShardReveal, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, verified_reveal) = msg.input;

            self.store.add_signed_reveal(&shard, &verified_reveal)?;

            self.shard_tracker
                .track_valid_reveal(shard, verified_reveal)
                .await?;

            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
