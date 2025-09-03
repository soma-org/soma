use std::{future::Future, sync::Arc, time::Duration};

use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::score_vote::{ScoreVote, ScoreVoteAPI},
};
use async_trait::async_trait;
use evaluation::EvaluationScore;
use fastcrypto::{bls12381::min_sig, traits::KeyPair};
use quick_cache::sync::{Cache, GuardResult};
use shared::{
    actors::{ActorHandle, ActorMessage, Processor},
    crypto::keys::{EncoderAggregateSignature, EncoderKeyPair, EncoderPublicKey, EncoderSignature},
    digest::Digest,
    error::{ShardError, ShardResult},
    scope::Scope,
    shard::Shard,
    signed::Signed,
    verified::Verified,
};
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};
use types::score_set::ScoreSetAPI;

use super::clean_up::CleanUpProcessor;

pub(crate) struct ScoreVoteProcessor<E: EncoderInternalNetworkClient> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<E>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    clean_up_pipeline: ActorHandle<CleanUpProcessor>,
    recv_dedup: Cache<(Digest<Shard>, EncoderPublicKey), ()>,
    send_dedup: Cache<Digest<Shard>, ()>,
}

impl<E: EncoderInternalNetworkClient> ScoreVoteProcessor<E> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        clean_up_pipeline: ActorHandle<CleanUpProcessor>,
        recv_cache_capacity: usize,
        send_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            broadcaster,
            encoder_keypair,
            clean_up_pipeline,
            recv_dedup: Cache::new(recv_cache_capacity),
            send_dedup: Cache::new(send_cache_capacity),
        }
    }
    pub async fn start_timer<F, Fut>(
        &self,
        timeout: Duration,
        cancellation: CancellationToken,
        on_trigger: F,
    ) where
        F: FnOnce() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ShardResult<()>> + Send + 'static,
    {
        tokio::spawn(async move {
            tokio::select! {
                _ = sleep(timeout) => {
                    on_trigger().await;
                }
                _ = cancellation.cancelled() => {
                    info!("skipping trigger for submitting on-chain and calling clean up pipeline");
                }
            }
        });
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient> Processor for ScoreVoteProcessor<E> {
    type Input = (Shard, Verified<ScoreVote>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, score_vote) = msg.input;
            let shard_digest = shard.digest()?;

            match self.recv_dedup.get_value_or_guard(
                &(shard_digest, score_vote.author().clone()),
                Some(Duration::from_secs(5)),
            ) {
                GuardResult::Value(_) => return Err(ShardError::RecvDuplicate),
                GuardResult::Guard(placeholder) => {
                    placeholder.insert(());
                }
                GuardResult::Timeout => (),
            }
            self.store.add_score_vote(&shard, &score_vote)?;
            info!(
                "Starting track_valid_scores for scorer: {:?}",
                score_vote.author()
            );

            let all_scores = self.store.get_score_vote(&shard)?;
            debug!(
                "Current score count: {}, quorum_threshold: {}",
                all_scores.len(),
                shard.quorum_threshold()
            );

            let matching_score_votes: Vec<ScoreVote> = all_scores
                .iter()
                .filter(|sv| {
                    score_vote.signed_score_set().winner() == sv.signed_score_set().winner()
                })
                .cloned()
                .collect();

            info!(
                "Found matching scores: {}, quorum_threshold: {}",
                matching_score_votes.len(),
                shard.quorum_threshold()
            );

            info!(
                "Matching scores: {:?}",
                matching_score_votes
                    .iter()
                    .map(|s| s.clone().signed_score_set().into_inner().score().clone())
                    .collect::<Vec<EvaluationScore>>()
            );

            if matching_score_votes.len() >= shard.quorum_threshold() as usize {
                match self
                    .send_dedup
                    .get_value_or_guard(&(shard_digest), Some(Duration::from_secs(5)))
                {
                    GuardResult::Value(_) => return Ok(()),
                    GuardResult::Guard(placeholder) => {
                        // TODO: replace this for all inserts? the error should be propogated or unwrapped
                        placeholder.insert(()).unwrap();
                    }
                    GuardResult::Timeout => (),
                };
                info!("QUORUM OF MATCHING SCORES - Aggregating signatures");

                let (signatures, evaluators): (Vec<EncoderSignature>, Vec<EncoderPublicKey>) = {
                    let mut sigs = Vec::new();
                    let mut evaluators = Vec::new();
                    for score_vote in matching_score_votes.iter() {
                        let sig = EncoderSignature::from_bytes(
                            &score_vote.signed_score_set().raw_signature(),
                        )
                        .map_err(ShardError::SignatureAggregationFailure)?;
                        sigs.push(sig);
                        evaluators.push(score_vote.author().clone());
                    }
                    (sigs, evaluators)
                };

                debug!(
                    "Creating aggregate signature with {} signatures from {} evaluators",
                    signatures.len(),
                    evaluators.len()
                );

                let agg = EncoderAggregateSignature::new(&signatures)
                    .map_err(ShardError::SignatureAggregationFailure)?;

                info!(
                    "Successfully created aggregate score with {} evaluators",
                    evaluators.len()
                );

                self.store
                    .add_aggregate_score(&shard, (agg.clone(), evaluators))?;

                info!(
                    "SHARD CONSENSUS COMPLETE - Aggregate score stored: {:?}",
                    agg
                );

                if score_vote.signed_score_set().winner() == &self.encoder_keypair.public() {
                    // call on-chain
                    info!("MOCK SUBMIT ON CHAIN");
                }

                self.clean_up_pipeline
                    .process(shard.clone(), msg.cancellation.clone())
                    .await?;
            } else {
                debug!(
                    "Not enough matching scores yet - waiting for more scores. Matching scores: {}, \
                    quorum_threshold: {}",
                    matching_score_votes.len(),
                    shard.quorum_threshold()
                );
            }

            info!("Completed track_valid_scores");
            Ok(())
        }
        .await;
        msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
