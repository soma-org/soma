use std::{ops::Deref, sync::Arc};

use crate::{
    actors::{ActorMessage, Processor},
    error::ShardResult,
    storage::datastore::Store,
    types::{
        encoder_committee::EncoderIndex,
        shard::Shard,
        shard_votes::{CommitRound, ShardVotes},
    },
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{digest::Digest, signed::Signed, verified::Verified};

pub(crate) struct CommitVotesProcessor {
    store: Arc<dyn Store>,
    own_index: EncoderIndex,
}

impl CommitVotesProcessor {
    pub(crate) fn new(store: Arc<dyn Store>, own_index: EncoderIndex) -> Self {
        Self { store, own_index }
    }
}

#[async_trait]
impl Processor for CommitVotesProcessor {
    type Input = (
        Shard,
        Digest<Shard>,
        Verified<Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, shard_ref, votes) = msg.input;
            let epoch = shard.epoch();
            let (total_finalized_slots, total_accepted_slots) = self.store.add_commit_vote(
                epoch,
                shard_ref,
                shard.clone(),
                votes.deref().to_owned().deref().to_owned(),
            )?;

            if total_finalized_slots == shard.inference_size() {
                if total_accepted_slots >= shard.minimum_inference_size() as usize {
                    if shard.inference_set().contains(&self.own_index) {
                        // trigger reveal if member of inference set
                    }
                    // TODO: figure out how rejections are accounted for and whether eval set needs to do anything
                } else {
                    // trigger cancellation, this shard cannot proceed
                }
            }
            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
