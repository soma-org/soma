use std::sync::Arc;

use crate::{
    actors::{ActorMessage, Processor},
    core::shard_tracker::ShardTracker,
    datastore::Store,
    error::ShardResult,
    messaging::EncoderInternalNetworkClient,
    types::{shard::Shard, shard_commit_votes::ShardCommitVotes},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use objects::storage::ObjectStorage;
use shared::{signed::Signed, verified::Verified};

pub(crate) struct CommitVotesProcessor<E: EncoderInternalNetworkClient, S: ObjectStorage> {
    store: Arc<dyn Store>,
    shard_tracker: ShardTracker<E, S>,
}

impl<E: EncoderInternalNetworkClient, S: ObjectStorage> CommitVotesProcessor<E, S> {
    pub(crate) fn new(store: Arc<dyn Store>, shard_tracker: ShardTracker<E, S>) -> Self {
        Self {
            store,
            shard_tracker,
        }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient, S: ObjectStorage> Processor for CommitVotesProcessor<E, S> {
    type Input = (
        Shard,
        Verified<Signed<ShardCommitVotes, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, votes) = msg.input;

            self.store.add_commit_votes(&shard, &votes)?;
            self.shard_tracker
                .track_valid_commit_votes(shard, votes)
                .await?;

            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
