use std::sync::Arc;

use crate::{
    actors::{workers::broadcaster::BroadcasterProcessor, ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    networking::messaging::EncoderInternalNetworkClient,
    storage::datastore::Store,
    types::{shard::Shard, shard_verifier::ShardAuthToken},
};
use async_trait::async_trait;
use shared::{digest::Digest, signed::Signed, verified::Verified};

pub(crate) struct EvaluationProcessor<E: EncoderInternalNetworkClient> {
    store: Arc<dyn Store>,
    broadcaster: ActorHandle<BroadcasterProcessor<E>>,
}

impl<E: EncoderInternalNetworkClient> EvaluationProcessor<E> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: ActorHandle<BroadcasterProcessor<E>>,
    ) -> Self {
        Self { store, broadcaster }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient> Processor for EvaluationProcessor<E> {
    type Input = (ShardAuthToken, Shard);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (auth_token, shard) = msg.input;
            let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
            let epoch = shard.epoch();
            Ok(())
            // perform evaluation
            // package scores
            // broadcast scores
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
