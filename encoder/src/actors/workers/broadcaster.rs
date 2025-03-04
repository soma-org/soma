use std::{collections::HashSet, sync::Arc};

use fastcrypto::traits::KeyPair;
use shared::{
    crypto::keys::EncoderKeyPair, digest::Digest, scope::Scope, signed::Signed, verified::Verified,
};
use tokio::sync::Semaphore;

use crate::{
    core::broadcaster::Broadcaster,
    networking::messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    storage::datastore::Store,
    types::{
        encoder_committee::{EncoderIndex, Epoch},
        shard::Shard,
        shard_verifier::ShardAuthToken,
        shard_votes::{CommitRound, RevealRound, ShardVotes, ShardVotesV1},
    },
};
use async_trait::async_trait;

use crate::actors::{ActorMessage, Processor};

pub(crate) struct BroadcasterProcessor<C: EncoderInternalNetworkClient> {
    semaphore: Arc<Semaphore>,
    broadcaster: Arc<Broadcaster<C>>,
    store: Arc<dyn Store>,
    own_index: EncoderIndex,
    encoder_keypair: Arc<EncoderKeyPair>,
}

impl<C: EncoderInternalNetworkClient> BroadcasterProcessor<C> {
    pub(crate) fn new(
        concurrency: usize,
        broadcaster: Broadcaster<C>,
        store: Arc<dyn Store>,
        own_index: EncoderIndex,
        encoder_keypair: Arc<EncoderKeyPair>,
    ) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(concurrency)),
            broadcaster: Arc::new(broadcaster),
            store,
            own_index,
            encoder_keypair,
        }
    }
}

pub(crate) enum BroadcastType {
    CommitVote(Epoch, Digest<Shard>),
    RevealVote(Epoch, Digest<Shard>),
}

#[async_trait]
impl<C: EncoderInternalNetworkClient> Processor for BroadcasterProcessor<C> {
    type Input = (ShardAuthToken, Shard, BroadcastType, Vec<EncoderIndex>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let broadcaster = self.broadcaster.clone();
        let store = self.store.clone();
        let keypair = self.encoder_keypair.inner().copy();
        let own_index = self.own_index.clone();
        if let Ok(permit) = self.semaphore.clone().acquire_owned().await {
            tokio::spawn(async move {
                let (auth_token, shard, input, peers) = msg.input;

                match input {
                    BroadcastType::CommitVote(epoch, shard_ref) => {
                        // TODO: look up rejects from store

                        let accepts = store.get_filled_certified_commit_slots(epoch, shard_ref);
                        let inference_set: HashSet<EncoderIndex> =
                            shard.inference_set().into_iter().collect();
                        let accepts_set: HashSet<EncoderIndex> = accepts.into_iter().collect();

                        let rejects: Vec<EncoderIndex> = inference_set
                            .difference(&accepts_set)
                            .cloned() // Clone the EncoderIndex values (since difference gives references)
                            .collect();

                        let votes: ShardVotes<CommitRound> =
                            ShardVotes::V1(ShardVotesV1::new(auth_token, own_index, rejects));
                        let signed_votes =
                            Signed::new(votes, Scope::ShardCommitVotes, &keypair.private())
                                .unwrap();

                        let verified = Verified::from_trusted(signed_votes).unwrap();
                        let result = broadcaster
                            .broadcast(verified, peers, |client, peer, verified_type| async move {
                                client
                                    .send_commit_votes(peer, &verified_type, MESSAGE_TIMEOUT)
                                    .await;
                                Ok(())
                            })
                            .await;
                        msg.sender.send(result);
                    }
                    BroadcastType::RevealVote(epoch, shard_ref) => {
                        let accepts = store.get_filled_reveal_slots(epoch, shard_ref);
                        let inference_set: HashSet<EncoderIndex> =
                            shard.inference_set().into_iter().collect();
                        let accepts_set: HashSet<EncoderIndex> = accepts.into_iter().collect();

                        let rejects: Vec<EncoderIndex> = inference_set
                            .difference(&accepts_set)
                            .cloned() // Clone the EncoderIndex values (since difference gives references)
                            .collect();

                        let votes: ShardVotes<RevealRound> =
                            ShardVotes::V1(ShardVotesV1::new(auth_token, own_index, rejects));
                        let signed_votes =
                            Signed::new(votes, Scope::ShardCommitVotes, &keypair.private())
                                .unwrap();

                        let verified = Verified::from_trusted(signed_votes).unwrap();
                        let result = broadcaster
                            .broadcast(verified, peers, |client, peer, verified_type| async move {
                                client
                                    .send_reveal_votes(peer, &verified_type, MESSAGE_TIMEOUT)
                                    .await;
                                Ok(())
                            })
                            .await;
                        msg.sender.send(result);
                    }
                }
                drop(permit);
            });
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
