use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    actors::{ActorHandle, ActorMessage, Processor},
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    error::ShardResult,
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        shard::Shard,
        shard_commit_votes::{ShardCommitVotes, ShardCommitVotesAPI},
        shard_reveal::{ShardReveal, ShardRevealV1},
    },
};
use async_trait::async_trait;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use objects::storage::ObjectStorage;
use probe::messaging::ProbeClient;
use shared::{
    crypto::{
        keys::{EncoderKeyPair, EncoderPublicKey},
        Aes256IV, Aes256Key, EncryptionKey,
    },
    scope::Scope,
    signed::Signed,
    verified::Verified,
};
use tracing::{debug, info, warn};

use super::reveal::RevealProcessor;

pub(crate) struct CommitVotesProcessor<
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
    P: ProbeClient,
> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<E>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    reveal_pipeline: ActorHandle<RevealProcessor<E, S, P>>,
}

impl<E: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient>
    CommitVotesProcessor<E, S, P>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        reveal_pipeline: ActorHandle<RevealProcessor<E, S, P>>,
    ) -> Self {
        Self {
            store,
            broadcaster,
            encoder_keypair,
            reveal_pipeline,
        }
    }
}

#[derive(Eq, PartialEq, PartialOrd, Ord, Hash, Clone, Copy)]
enum Finality {
    Accepted,
    Rejected,
}
#[async_trait]
impl<E: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient> Processor
    for CommitVotesProcessor<E, S, P>
{
    type Input = (
        Shard,
        Verified<Signed<ShardCommitVotes, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, commit_votes) = msg.input;

            self.store.add_commit_votes(&shard, &commit_votes)?;
            info!(
                "Starting track_valid_commit_votes for voter: {:?}",
                commit_votes.voter()
            );
            let mut finalized_encoders: HashMap<EncoderPublicKey, Finality> = HashMap::new();
            let num_votes = self.store.count_commit_votes(&shard)?;
            debug!("Current commit vote count: {}", num_votes);
            let accepts_keys: HashSet<_> = commit_votes
                .accepts()
                .into_iter()
                .map(|(key, _)| key.clone())
                .collect();
            let encoders_set: HashSet<_> = shard.encoders().into_iter().collect();
            let rejects: Vec<EncoderPublicKey> =
                encoders_set.difference(&accepts_keys).cloned().collect();
            let remaining_votes = shard.size() - num_votes;
            debug!(
                "Processing votes breakdown - accepts: {}, rejects: {}, remaining: {}",
                accepts_keys.len(),
                rejects.len(),
                remaining_votes
            );
            for (encoder, digest) in commit_votes.accepts() {
                let vote_counts =
                    self.store
                        .get_commit_votes_for_encoder(&shard, encoder, Some(digest))?;
                // debug!("Evaluating encoder commit votes - encoder: {:?}, accept_count: {:?}, reject_count: {}, highest: {}, quorum_threshold: {}",
                //    encoder, vote_counts.accept_count().unwrap_or(0_usize), vote_counts.reject_count(),
                //    vote_counts.highest(), shard.quorum_threshold());
                if vote_counts.accept_count().unwrap_or(0_usize)
                    >= shard.quorum_threshold() as usize
                {
                    info!(
                        "Encoder commit ACCEPTED - reached quorum threshold for encoder: {:?}",
                        encoder
                    );
                    finalized_encoders.insert(encoder.clone(), Finality::Accepted);
                } else if vote_counts.reject_count() >= shard.rejection_threshold() as usize
                    || vote_counts.highest() + remaining_votes < shard.quorum_threshold() as usize
                {
                    info!(
                        "Encoder commit REJECTED - either reject count >= quorum or can't reach \
                         quorum for encoder: {:?}",
                        encoder
                    );
                    finalized_encoders.insert(encoder.clone(), Finality::Rejected);
                } else {
                    debug!(
                        "Encoder commit still PENDING - not enough votes yet for encoder: {:?}",
                        encoder
                    );
                }
            }
            for encoder in rejects {
                let vote_counts = self
                    .store
                    .get_commit_votes_for_encoder(&shard, &encoder, None)?;
                debug!(
                    "Evaluating rejected encoder - encoder: {:?}, reject_count: {}, highest: {}, \
                     quorum_threshold: {}",
                    encoder,
                    vote_counts.reject_count(),
                    vote_counts.highest(),
                    shard.quorum_threshold()
                );
                if vote_counts.reject_count() >= shard.rejection_threshold() as usize
                    || vote_counts.highest() + remaining_votes < shard.quorum_threshold() as usize
                {
                    info!(
                        "Rejected encoder commit REJECTED - either reject count >= quorum or \
                         can't reach quorum for encoder: {:?}",
                        encoder
                    );
                    finalized_encoders.insert(encoder.clone(), Finality::Rejected);
                } else {
                    debug!(
                        "Rejected encoder still PENDING - not enough votes yet for encoder: {:?}",
                        encoder
                    );
                }
            }
            let total_finalized = finalized_encoders.len();
            let total_accepted = finalized_encoders
                .values()
                .filter(|&&f| f == Finality::Accepted)
                .count();
            info!(
                "Finality status for commit votes - total_finalized: {}, total_accepted: {}, \
                 shard_size: {}, quorum_threshold: {}",
                total_finalized,
                total_accepted,
                shard.size(),
                shard.quorum_threshold()
            );
            if total_finalized == shard.size() {
                if total_accepted >= shard.quorum_threshold() as usize {
                    // let already_revealing = self.store.count_signed_reveal(&shard).unwrap_or(0) > 0;
                    // if already_revealing {
                    //     info!(
                    //         "Skipping trigger of BroadcastAction::Reveal as reveal phase has \
                    //          already started"
                    //     );
                    //     return Ok(());
                    // }
                    info!("ALL COMMIT VOTES FINALIZED - Proceeding to REVEAL phase");
                    info!("Beginning to broadcast reveal");
                    // Generate key from signature over shard
                    let inner_keypair = self.encoder_keypair.inner().copy();
                    let signed_shard = Signed::new(
                        shard.clone(),
                        Scope::EncryptionKey,
                        &inner_keypair.copy().private(),
                    )
                    .unwrap();
                    let signature_bytes = signed_shard.raw_signature();
                    let mut key_bytes = [0u8; 32];
                    key_bytes.copy_from_slice(&signature_bytes[..32]); // Use only the first 32 bytes
                    let mut iv_bytes = [0u8; 16];
                    iv_bytes.copy_from_slice(&signature_bytes[..16]); // Use only the first 16 bytes
                    let key = EncryptionKey::Aes256(Aes256IV {
                        iv: iv_bytes,
                        key: Aes256Key::from(key_bytes),
                    });
                    // Create reveal
                    let reveal = ShardReveal::V1(ShardRevealV1::new(
                        commit_votes.auth_token().clone(),
                        self.encoder_keypair.public(),
                        key,
                    ));
                    // Sign reveal
                    let signed_reveal =
                        Signed::new(reveal, Scope::ShardReveal, &inner_keypair.private()).unwrap();
                    let verified = Verified::from_trusted(signed_reveal).unwrap();
                    info!("Broadcasting reveal to other nodes");
                    // call reveal pipeline
                    self.reveal_pipeline
                        .process((shard.clone(), verified.clone()), msg.cancellation.clone())
                        .await?;
                    // Broadcast to other encoders
                    self.broadcaster
                        .broadcast(
                            verified.clone(),
                            shard.encoders(),
                            |client, peer, verified_type| async move {
                                client
                                    .send_reveal(&peer, &verified_type, MESSAGE_TIMEOUT)
                                    .await?;
                                Ok(())
                            },
                        )
                        .await?;
                    // broadcast reveal
                } else {
                    warn!(
                        "ALL COMMIT VOTES FINALIZED - But failed to reach quorum, shard will be \
                         terminated. Total accepted: {}, quorum_threshold: {}",
                        total_accepted,
                        shard.quorum_threshold()
                    );
                    // should clean up and shutdown the shard since it will not be able to complete
                }
            } else {
                debug!(
                    "Not all commits finalized yet - continuing to collect votes. Total \
                     finalized: {}, shard_size: {}",
                    total_finalized,
                    shard.size()
                );
            }
            info!("Completed track_valid_commit_votes");
            Ok(())
        }
        .await;
        msg.sender.send(result);
    }
    fn shutdown(&mut self) {}
}
