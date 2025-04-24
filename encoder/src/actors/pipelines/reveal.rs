use crate::{
    actors::{ActorMessage, Processor},
    core::shard_tracker::ShardTracker,
    datastore::Store,
    error::ShardResult,
    messaging::EncoderInternalNetworkClient,
    types::{shard::Shard, shard_reveal::ShardReveal},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use objects::storage::ObjectStorage;
use shared::{signed::Signed, verified::Verified};
use std::sync::Arc;

pub(crate) struct RevealProcessor<E: EncoderInternalNetworkClient, S: ObjectStorage> {
    store: Arc<dyn Store>,
    shard_tracker: Arc<ShardTracker<E, S>>,
}

impl<E: EncoderInternalNetworkClient, S: ObjectStorage> RevealProcessor<E, S> {
    pub(crate) fn new(store: Arc<dyn Store>, shard_tracker: Arc<ShardTracker<E, S>>) -> Self {
        Self {
            store,
            shard_tracker,
        }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient, S: ObjectStorage> Processor for RevealProcessor<E, S> {
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
