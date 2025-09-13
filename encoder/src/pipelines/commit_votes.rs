use std::{collections::HashMap, sync::Arc};

use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::{ShardStage, Store},
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        commit_votes::{CommitVotes, CommitVotesAPI},
        reveal::{Reveal, RevealV1},
    },
};
use async_trait::async_trait;
use evaluation::messaging::EvaluationClient;
use objects::{networking::ObjectNetworkClient, storage::ObjectStorage};
use tracing::{debug, info, warn};
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::ShardResult,
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        keys::{EncoderKeyPair, EncoderPublicKey},
        verified::Verified,
    },
    submission::Submission,
};

use super::reveal::RevealProcessor;

pub(crate) struct CommitVotesProcessor<
    O: ObjectNetworkClient,
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
    P: EvaluationClient,
> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<E>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    reveal_pipeline: ActorHandle<RevealProcessor<O, E, S, P>>,
}

impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > CommitVotesProcessor<O, E, S, P>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        reveal_pipeline: ActorHandle<RevealProcessor<O, E, S, P>>,
    ) -> Self {
        Self {
            store,
            broadcaster,
            encoder_keypair,
            reveal_pipeline,
        }
    }
}

#[async_trait]
impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > Processor for CommitVotesProcessor<O, E, S, P>
{
    type Input = (Shard, Verified<CommitVotes>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, commit_votes) = msg.input;
            let author = commit_votes.author();

            // Store the stage message and the commit votes (no aggregation in store).
            self.store.add_shard_stage_message(
                &shard,
                ShardStage::CommitVote,
                &author,
            )?;
            self.store.add_commit_votes(&shard, &commit_votes)?;

            info!("Processing commit votes from voter: {:?}", author);

            let all_votes = self.store.get_all_commit_votes(&shard)?;
            let num_votes = all_votes.len();
            debug!("Current commit vote count: {}", num_votes);

            let remaining_votes = shard.size() - num_votes;

            let mut accept_counts: HashMap<EncoderPublicKey, HashMap<Digest<Submission>, usize>> = HashMap::new();

            for votes in &all_votes {
                for (target, digest) in votes.accepts() {
                    let encoder_entry = accept_counts.entry(target.clone()).or_insert_with(HashMap::new);
                    *encoder_entry.entry(digest.clone()).or_insert(0) += 1;
                }
            }

            // Map to store finalized encoders and their accepted digest (or None if rejected)
            let mut finalized_encoders: HashMap<EncoderPublicKey, Option<Digest<Submission>>> = HashMap::new();

            for encoder in shard.encoders() {
                let accept_map = accept_counts.get(&encoder).cloned().unwrap_or_default();
                let highest = accept_map.values().cloned().max().unwrap_or(0);
                let sum_accept_votes: usize = accept_map.values().sum();
                let reject_count = num_votes - sum_accept_votes;

                debug!(
                    "Evaluating encoder: {:?}, highest: {}, sum_accepts: {}, rejects: {}, remaining: {}",
                    encoder, highest, sum_accept_votes, reject_count, remaining_votes
                );

                if highest >= shard.quorum_threshold() as usize {
                    // Find the digest with the highest vote count
                    let accepted_digest = accept_map
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(digest, _)| digest);

                    info!(
                        "Encoder commit ACCEPTED - reached quorum for encoder: {:?} with digest: {:?}", 
                        encoder, accepted_digest
                    );
                    finalized_encoders.insert(encoder.clone(), accepted_digest);
                } else if reject_count >= shard.rejection_threshold() as usize
                    || highest + remaining_votes < shard.quorum_threshold() as usize
                {
                    info!(
                        "Encoder commit REJECTED - reject count >= rejection threshold or can't reach quorum for encoder: {:?}",
                        encoder
                    );
                    finalized_encoders.insert(encoder.clone(), None);
                } else {
                    debug!("Encoder commit PENDING - insufficient votes for encoder: {:?}", encoder);
                }
            }

            let total_finalized = finalized_encoders.len();
            let total_accepted = finalized_encoders
                .values()
                .filter(|digest| digest.is_some())
                .count();
            info!(
                "Finality status - total_finalized: {}, total_accepted: {}, shard_size: {}, quorum_threshold: {}",
                total_finalized,
                total_accepted,
                shard.size(),
                shard.quorum_threshold()
            );

            if total_finalized == shard.size() {
                if total_accepted >= shard.quorum_threshold() as usize {
                    self.store
                        .add_shard_stage_dispatch(&shard, ShardStage::Reveal)?;
                    info!("ALL COMMIT VOTES FINALIZED - Proceeding to REVEAL phase");

                    // Log the accepted commits in the store
                    for (encoder, digest) in &finalized_encoders {
                        if let Some(digest) = digest {
                            self.store.add_accepted_commit(&shard, encoder, digest.clone())?;
                            info!("Logged accepted commit digest for encoder: {:?}", encoder);
                        }
                    }

                    let own_key = self.encoder_keypair.public();
                    if let Some(Some(own_submission_digest)) = finalized_encoders.get(&own_key) {
                        let (own_submission, _) = self.store.get_submission(&shard, own_submission_digest.clone())?;

                        let own_reveal = Reveal::V1(RevealV1::new(
                            commit_votes.auth_token().clone(),
                            own_key,
                            own_submission,
                        ));
                        let verified_reveal = Verified::from_trusted(own_reveal).unwrap();
                        info!("Broadcasting own reveal to other nodes");

                        // Call reveal pipeline.
                        self.reveal_pipeline
                            .process(
                                (shard.clone(), verified_reveal.clone()),
                                msg.cancellation.clone(),
                            )
                            .await?;

                        // Broadcast to other encoders.
                        self.broadcaster
                            .broadcast(
                                verified_reveal.clone(),
                                shard.encoders(),
                                |client, peer, verified_type| async move {
                                    client
                                        .send_reveal(&peer, &verified_type, MESSAGE_TIMEOUT)
                                        .await?;
                                    Ok(())
                                },
                            )
                            .await?;
                    } else {
                        info!("Own encoder not accepted; skipping reveal broadcast.");
                    }
                } else {
                    warn!(
                        "ALL COMMIT VOTES FINALIZED - Failed to reach quorum; shard terminating. Total accepted: {}, quorum_threshold: {}",
                        total_accepted,
                        shard.quorum_threshold()
                    );
                    // Should clean up and shutdown the shard since it will not be able to complete.
                }
            } else {
                debug!(
                    "Collecting more votes - total finalized: {}, shard_size: {}",
                    total_finalized,
                    shard.size()
                );
            }

            info!("Completed processing commit votes");
            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
