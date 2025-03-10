use std::{ops::Deref, sync::Arc};

use crate::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    networking::messaging::EncoderInternalNetworkClient,
    storage::{datastore::Store, object::ObjectStorage},
    types::{
        encoder_committee::EncoderIndex,
        shard::Shard,
        shard_verifier::ShardAuthToken,
        shard_votes::{RevealRound, ShardVotes},
    },
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{digest::Digest, signed::Signed, verified::Verified};

use super::evaluation::EvaluationProcessor;

pub(crate) struct RevealVotesProcessor<E: EncoderInternalNetworkClient, S: ObjectStorage> {
    store: Arc<dyn Store>,
    own_index: EncoderIndex,
    evaluation_handle: ActorHandle<EvaluationProcessor<E, S>>,
}

impl<E: EncoderInternalNetworkClient, S: ObjectStorage> RevealVotesProcessor<E, S> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        own_index: EncoderIndex,
        evaluation_handle: ActorHandle<EvaluationProcessor<E, S>>,
    ) -> Self {
        Self {
            store,
            own_index,
            evaluation_handle,
        }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient, S: ObjectStorage> Processor for RevealVotesProcessor<E, S> {
    type Input = (
        ShardAuthToken,
        Shard,
        Verified<Signed<ShardVotes<RevealRound>, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (auth_token, shard, votes) = msg.input;
            let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
            let epoch = shard.epoch();
            let (total_finalized_slots, total_accepted_slots) = self.store.add_reveal_vote(
                epoch,
                shard_ref,
                shard.clone(),
                votes.deref().to_owned().deref().to_owned(),
            )?;

            if total_finalized_slots == shard.inference_size() {
                if total_accepted_slots >= shard.minimum_inference_size() as usize {
                    if shard.evaluation_set().contains(&self.own_index) {
                        self.evaluation_handle
                            .process((auth_token, shard), msg.cancellation.clone())
                            .await?;
                    }
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
