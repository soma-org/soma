use std::{collections::HashMap, sync::Arc, time::Duration};

use crate::{
    datastore::Store,
    messaging::EncoderInternalNetworkClient,
    types::shard_reveal_votes::{ShardRevealVotes, ShardRevealVotesAPI},
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use objects::storage::ObjectStorage;
use probe::messaging::ProbeClient;
use quick_cache::sync::{Cache, GuardResult};
use shared::{
    actors::{ActorHandle, ActorMessage, Processor},
    crypto::keys::EncoderPublicKey,
    digest::Digest,
    error::{ShardError, ShardResult},
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use tracing::{debug, error, info, warn};

use super::evaluation::EvaluationProcessor;

pub(crate) struct RevealVotesProcessor<
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
    P: ProbeClient,
> {
    store: Arc<dyn Store>,
    evaluation_pipeline: ActorHandle<EvaluationProcessor<E, S, P>>,
    recv_dedup: Cache<(Digest<Shard>, EncoderPublicKey), ()>,
    send_dedup: Cache<Digest<Shard>, ()>,
}

impl<E: EncoderInternalNetworkClient, S: ObjectStorage, P: ProbeClient>
    RevealVotesProcessor<E, S, P>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        evaluation_pipeline: ActorHandle<EvaluationProcessor<E, S, P>>,
        recv_cache_capacity: usize,
        send_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            evaluation_pipeline,
            recv_dedup: Cache::new(recv_cache_capacity),
            send_dedup: Cache::new(send_cache_capacity),
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
    for RevealVotesProcessor<E, S, P>
{
    type Input = (
        Shard,
        Verified<Signed<ShardRevealVotes, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, reveal_votes) = msg.input;
            let shard_digest = shard.digest()?;

            match self.recv_dedup.get_value_or_guard(
                &(shard_digest, reveal_votes.voter().clone()),
                Some(Duration::from_secs(5)),
            ) {
                GuardResult::Value(_) => return Err(ShardError::RecvDuplicate),
                GuardResult::Guard(placeholder) => {
                    placeholder.insert(());
                }
                GuardResult::Timeout => (),
            }
            self.store.add_reveal_votes(&shard, &reveal_votes)?;
            info!(
                "Starting track_valid_reveal_votes for voter: {:?}",
                reveal_votes.voter()
            );

            let mut finalized_encoders: HashMap<EncoderPublicKey, Finality> = HashMap::new();

            // Log each encoder's vote status
            for encoder in shard.encoders() {
                let agg_votes = self.store.get_reveal_votes_for_encoder(&shard, &encoder)?;

                debug!(
                "Evaluating reveal votes for encoder: {:?}, accept_count: {}, reject_count: {}, \
                 quorum_threshold: {}, rejection_threshold: {}",
                encoder,
                agg_votes.accept_count(),
                agg_votes.reject_count(),
                shard.quorum_threshold(),
                shard.rejection_threshold()
            );

                if agg_votes.accept_count() >= shard.quorum_threshold() as usize {
                    info!(
                        "Encoder reveal ACCEPTED - reached quorum threshold for encoder: {:?}",
                        encoder
                    );
                    finalized_encoders.insert(encoder.clone(), Finality::Accepted);
                } else if agg_votes.reject_count() >= shard.rejection_threshold() as usize {
                    info!(
                        "Encoder reveal REJECTED - reached rejection threshold for encoder: {:?}",
                        encoder
                    );
                    finalized_encoders.insert(encoder.clone(), Finality::Rejected);
                } else {
                    debug!(
                        "Encoder reveal still PENDING - not enough votes yet for encoder: {:?}",
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
                "Finality status for reveal votes - total_finalized: {}, total_accepted: {}, \
             shard_size: {}, quorum_threshold: {}",
                total_finalized,
                total_accepted,
                shard.size(),
                shard.quorum_threshold()
            );

            if total_finalized == shard.size() {
                if total_accepted >= shard.quorum_threshold() as usize {
                    match self
                        .send_dedup
                        .get_value_or_guard(&(shard_digest), Some(Duration::from_secs(5)))
                    {
                        GuardResult::Value(_) => return Ok(()),
                        GuardResult::Guard(placeholder) => {
                            placeholder.insert(());
                        }
                        GuardResult::Timeout => (),
                    };
                    info!("ALL REVEAL VOTES FINALIZED - Proceeding to EVALUATION phase");
                    let input = (reveal_votes.auth_token().to_owned(), shard.clone());
                    info!("Triggering evaluation process");
                    self.evaluation_pipeline
                        .process(input, msg.cancellation.clone())
                        .await?;
                    info!("Evaluation process successfully triggered");
                } else {
                    warn!(
                        "ALL REVEAL VOTES FINALIZED - But failed to reach quorum, shard will be \
                     terminated. Total accepted: {}, quorum_threshold: {}",
                        total_accepted,
                        shard.quorum_threshold()
                    );
                    // should clean up and shutdown the shard since it will not be able to complete
                }
            } else {
                debug!(
                "Not all reveals finalized yet - continuing to collect votes. Total finalized: \
                 {}, shard_size: {}",
                total_finalized,
                shard.size()
            );
            }

            info!("Completed track_valid_reveal_votes");
            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
