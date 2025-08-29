use std::{future::Future, marker::PhantomData, sync::Arc, time::Duration};

use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::Store,
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::finality::{Finality, FinalityV1},
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
use types::shard_score::{ScoreSetAPI, ShardScore, ShardScoreAPI};

use super::finality::FinalityProcessor;

pub(crate) struct ScoresProcessor<E: EncoderInternalNetworkClient> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<E>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    finality_pipeline: ActorHandle<FinalityProcessor>,
    recv_dedup: Cache<(Digest<Shard>, EncoderPublicKey), ()>,
    send_dedup: Cache<Digest<Shard>, ()>,
}

impl<E: EncoderInternalNetworkClient> ScoresProcessor<E> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        finality_pipeline: ActorHandle<FinalityProcessor>,
        recv_cache_capacity: usize,
        send_cache_capacity: usize,
    ) -> Self {
        Self {
            store,
            broadcaster,
            encoder_keypair,
            finality_pipeline,
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
impl<E: EncoderInternalNetworkClient> Processor for ScoresProcessor<E> {
    type Input = (
        Shard,
        Verified<Signed<ShardScore, min_sig::BLS12381Signature>>,
    );
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, scores) = msg.input;
            let shard_digest = shard.digest()?;

            match self.recv_dedup.get_value_or_guard(
                &(shard_digest, scores.author().clone()),
                Some(Duration::from_secs(5)),
            ) {
                GuardResult::Value(_) => return Err(ShardError::RecvDuplicate),
                GuardResult::Guard(placeholder) => {
                    placeholder.insert(());
                }
                GuardResult::Timeout => (),
            }
            self.store.add_signed_score(&shard, &scores)?;
            info!(
                "Starting track_valid_scores for scorer: {:?}",
                scores.author()
            );

            let all_scores = self.store.get_signed_scores(&shard)?;
            debug!(
                "Current score count: {}, quorum_threshold: {}",
                all_scores.len(),
                shard.quorum_threshold()
            );

            let matching_scores: Vec<Signed<ShardScore, min_sig::BLS12381Signature>> = all_scores
                .iter()
                .filter(|score| {
                    score.signed_score_set().winner() == scores.signed_score_set().winner()
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
                        .score()
                        .clone())
                    .collect::<Vec<EvaluationScore>>()
            );

            if matching_scores.len() >= shard.quorum_threshold() as usize {
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
                    for signed_scores in matching_scores.iter() {
                        let sig = EncoderSignature::from_bytes(&signed_scores.raw_signature())
                            .map_err(ShardError::SignatureAggregationFailure)?;
                        sigs.push(sig);
                        evaluators.push(signed_scores.author().clone());
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

                if scores.signed_score_set().winner() == &self.encoder_keypair.public() {
                    let finality_pipeline = self.finality_pipeline.clone();
                    let auth_token = scores.auth_token().clone();
                    let broadcaster = self.broadcaster.clone();
                    let encoder_keypair = self.encoder_keypair.clone();
                    let cancellation = msg.cancellation.clone();
                    // call on-chain
                    info!("MOCK SUBMIT ON CHAIN");
                    let finality =
                        Finality::V1(FinalityV1::new(auth_token, encoder_keypair.public()));
                    let inner_keypair = encoder_keypair.inner().copy();

                    // Sign scores
                    let signed_finality =
                        Signed::new(finality, Scope::Finality, &inner_keypair.private()).unwrap();
                    let verified = Verified::from_trusted(signed_finality).unwrap();

                    info!("dispatching finality pipeline to oneself");
                    finality_pipeline
                        .process((shard.clone(), verified.clone()), cancellation.clone())
                        .await?;

                    info!("broadcasting finality pipeline to peers");
                    broadcaster
                        .broadcast(
                            verified.clone(),
                            shard.encoders(),
                            |client, peer, verified_type| async move {
                                client
                                    .send_finality(&peer, &verified_type, MESSAGE_TIMEOUT)
                                    .await?;
                                Ok(())
                            },
                        )
                        .await?;

                    // broadcast to peers
                }
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
