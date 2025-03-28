use std::{collections::HashSet, ops::Deref, sync::Arc};

use crate::{
    actors::{
        workers::broadcaster::{BroadcastType, BroadcasterProcessor},
        ActorHandle, ActorMessage, Processor,
    },
    error::{ShardError, ShardResult},
    messaging::tonic::internal::EncoderInternalTonicClient,
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
use shared::{crypto::keys::EncoderPublicKey, digest::Digest, signed::Signed, verified::Verified};

pub(crate) struct CommitVotesProcessor {
    store: Arc<dyn Store>,
    broadcaster: ActorHandle<BroadcasterProcessor<EncoderInternalTonicClient>>,
    own_encoder_key: Arc<EncoderPublicKey>,
}

impl CommitVotesProcessor {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: ActorHandle<BroadcasterProcessor<EncoderInternalTonicClient>>,
        own_encoder_key: Arc<EncoderPublicKey>,
    ) -> Self {
        Self {
            store,
            broadcaster,
            own_encoder_key,
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
            let epoch = auth_token.epoch();
            let (total_finalized_slots, total_accepted_slots) = self.store.add_commit_vote(
                epoch,
                shard_ref,
                shard.clone(),
                votes.deref().to_owned().deref().to_owned(),
            )?;

            if total_finalized_slots == shard.inference_size() {
                if total_accepted_slots >= shard.minimum_inference_size() as usize {
                    if shard.inference_set_contains(&self.own_encoder_key) {
                        let peers = shard.shard_set();
                        let _ = self
                            .broadcaster
                            .process(
                                (
                                    auth_token,
                                    shard,
                                    BroadcastType::RevealKey(epoch, shard_ref),
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
