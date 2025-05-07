use std::{marker::PhantomData, sync::Arc, time::Duration};

use crate::{
    actors::{ActorMessage, Processor},
    datastore::Store,
    error::{ShardError, ShardResult},
    messaging::EncoderInternalNetworkClient,
    types::{
        shard::Shard,
        shard_scores::{Score, ScoreSetAPI, ShardScores, ShardScoresAPI},
    },
};
use async_trait::async_trait;
use fastcrypto::bls12381::min_sig;
use quick_cache::sync::{Cache, GuardResult};
use shared::{
    crypto::keys::{EncoderAggregateSignature, EncoderPublicKey, EncoderSignature},
    digest::Digest,
    signed::Signed,
    verified::Verified,
};
use tracing::{debug, info};

pub(crate) struct ScoresProcessor<E: EncoderInternalNetworkClient> {
    store: Arc<dyn Store>,
    marker: PhantomData<E>,
    recv_dedup: Cache<(Digest<Shard>, EncoderPublicKey), ()>,
    send_dedup: Cache<Digest<Shard>, ()>,
}

impl<E: EncoderInternalNetworkClient> ScoresProcessor<E> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        recv_cache_capacity: usize,
        send_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            marker: PhantomData,
            recv_dedup: Cache::new(recv_cache_capacity),
            send_dedup: Cache::new(send_cache_capacity),
        }
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient> Processor for ScoresProcessor<E> {
    type Input = (
        Shard,
        Verified<Signed<ShardScores, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, scores) = msg.input;
            let shard_digest = shard.digest()?;

            match self.recv_dedup.get_value_or_guard(
                &(shard_digest, scores.evaluator().clone()),
                Some(Duration::from_secs(5)),
            ) {
                GuardResult::Value(_) => return Err(ShardError::RecvDuplicate),
                GuardResult::Guard(placeholder) => {
                    placeholder.insert(());
                }
                GuardResult::Timeout => (),
            }
            self.store.add_signed_scores(&shard, &scores)?;
            info!(
                "Starting track_valid_scores for scorer: {:?}",
                scores.evaluator()
            );

            let all_scores = self.store.get_signed_scores(&shard)?;
            debug!(
                "Current score count: {}, quorum_threshold: {}",
                all_scores.len(),
                shard.quorum_threshold()
            );

            let matching_scores: Vec<Signed<ShardScores, min_sig::BLS12381Signature>> = all_scores
                .iter()
                .filter(|score| {
                    score.signed_score_set().into_inner() == scores.signed_score_set().into_inner()
                })
                .cloned()
                .collect();

            info!(
                "Found matching scores: {}, quorum_threshold: {}",
                matching_scores.len(),
                shard.quorum_threshold()
            );

            info!(
                "Matching scores: {:?}",
                matching_scores
                    .iter()
                    .map(|s| s
                        .clone()
                        .into_inner()
                        .signed_score_set()
                        .into_inner()
                        .scores())
                    .collect::<Vec<Vec<Score>>>()
            );

            if matching_scores.len() >= shard.quorum_threshold() as usize {
                info!("QUORUM OF MATCHING SCORES - Aggregating signatures");

                let (signatures, evaluators): (Vec<EncoderSignature>, Vec<EncoderPublicKey>) = {
                    let mut sigs = Vec::new();
                    let mut evaluators = Vec::new();
                    for signed_scores in matching_scores.iter() {
                        let sig = EncoderSignature::from_bytes(&signed_scores.raw_signature())
                            .map_err(ShardError::SignatureAggregationFailure)?;
                        sigs.push(sig);
                        evaluators.push(signed_scores.evaluator().clone());
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
            } else {
                debug!(
                "Not enough matching scores yet - waiting for more scores. Matching scores: {}, \
                 quorum_threshold: {}",
                matching_scores.len(),
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
