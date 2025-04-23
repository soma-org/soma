use std::sync::Arc;

use crate::{
    actors::{ActorMessage, Processor},
    core::shard_tracker::ShardTracker,
    datastore::Store,
    error::ShardResult,
    messaging::EncoderInternalNetworkClient,
    types::{shard::Shard, shard_scores::ShardScores},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use objects::storage::ObjectStorage;
use shared::{signed::Signed, verified::Verified};

pub(crate) struct ScoresProcessor<E: EncoderInternalNetworkClient, S: ObjectStorage> {
    store: Arc<dyn Store>,
    shard_tracker: ShardTracker<E, S>,
}

impl<E: EncoderInternalNetworkClient, S: ObjectStorage> ScoresProcessor<E, S> {
    pub(crate) fn new(store: Arc<dyn Store>, shard_tracker: ShardTracker<E, S>) -> Self {
        Self {
            store,
            shard_tracker,
        }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient, S: ObjectStorage> Processor for ScoresProcessor<E, S> {
    type Input = (
        Shard,
        Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, scores) = msg.input;
            self.store.add_signed_scores(&shard, &scores)?;
            self.shard_tracker.track_valid_scores(shard, scores).await?;
            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
