use std::sync::Arc;

use crate::{
    actors::{ActorMessage, Processor},
    core::shard_tracker::ShardTracker,
    datastore::Store,
    error::ShardResult,
    types::{shard::Shard, shard_reveal_votes::ShardRevealVotes},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{signed::Signed, verified::Verified};

pub(crate) struct RevealVotesProcessor {
    store: Arc<dyn Store>,
    shard_tracker: ShardTracker,
}

impl RevealVotesProcessor {
    pub(crate) fn new(store: Arc<dyn Store>, shard_tracker: ShardTracker) -> Self {
        Self {
            store,
            shard_tracker,
        }
    }
}

#[async_trait]
impl Processor for RevealVotesProcessor {
    type Input = (
        Shard,
        Verified<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, votes) = msg.input;
            self.store.add_reveal_votes(&shard, &votes)?;
            self.shard_tracker
                .track_valid_reveal_votes(shard, votes)
                .await?;

            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
