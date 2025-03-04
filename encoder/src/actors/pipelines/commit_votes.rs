use std::{collections::HashSet, ops::Deref, sync::Arc};

use crate::{
    actors::{
        workers::broadcaster::{BroadcastType, BroadcasterProcessor},
        ActorHandle, ActorMessage, Processor,
    },
    error::{ShardError, ShardResult},
    networking::messaging::tonic_network::EncoderInternalTonicClient,
    storage::datastore::Store,
    types::{
        encoder_committee::EncoderIndex,
        shard::Shard,
        shard_verifier::ShardAuthToken,
        shard_votes::{CommitRound, ShardVotes},
    },
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use shared::{digest::Digest, signed::Signed, verified::Verified};

pub(crate) struct CommitVotesProcessor {
    store: Arc<dyn Store>,
    own_index: EncoderIndex,
    broadcaster: ActorHandle<BroadcasterProcessor<EncoderInternalTonicClient>>,
}

impl CommitVotesProcessor {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        own_index: EncoderIndex,
        broadcaster: ActorHandle<BroadcasterProcessor<EncoderInternalTonicClient>>,
    ) -> Self {
        Self {
            store,
            own_index,
            broadcaster,
        }
    }
}

#[async_trait]
impl Processor for CommitVotesProcessor {
    type Input = (
        ShardAuthToken,
        Shard,
        Verified<Signed<ShardVotes<CommitRound>, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (auth_token, shard, votes) = msg.input;
            let shard_ref = Digest::new(&shard).map_err(ShardError::DigestFailure)?;
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
                        let inference_set = shard.inference_set(); // Vec<EncoderIndex>
                        let evaluation_set = shard.evaluation_set(); // Vec<EncoderIndex>

                        // Combine into a HashSet to deduplicate
                        let mut peers_set: HashSet<EncoderIndex> =
                            inference_set.into_iter().collect();
                        peers_set.extend(evaluation_set);

                        // Convert back to Vec
                        let peers: Vec<EncoderIndex> = peers_set.into_iter().collect();
                        let _ = self
                            .broadcaster
                            .process(
                                (
                                    auth_token,
                                    shard,
                                    BroadcastType::RevealKey(epoch, shard_ref, self.own_index),
                                    peers,
                                ),
                                msg.cancellation.clone(),
                            )
                            .await;
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
