use std::{future::Future, ops::Deref, sync::Arc, time::Duration};

use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::{ShardStage, Store},
    messaging::EncoderInternalNetworkClient,
    types::report_vote::{ReportVote, ReportVoteAPI},
};
use async_trait::async_trait;
use tokio::sync::RwLock;
use sdk::wallet_context::{self, WalletContext};
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    report::ReportAPI,
    shard::Shard,
    shard_crypto::{
        keys::{EncoderAggregateSignature, EncoderKeyPair, EncoderPublicKey, EncoderSignature},
        verified::Verified,
    },
    submission::SubmissionAPI,
    transaction::{TransactionData, TransactionKind},
};

use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use super::clean_up::CleanUpProcessor;

pub(crate) struct ReportVoteProcessor<E: EncoderInternalNetworkClient> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<E>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    clean_up_pipeline: ActorHandle<CleanUpProcessor>,
    wallet_context: Arc<RwLock<WalletContext>>,
}

impl<E: EncoderInternalNetworkClient> ReportVoteProcessor<E> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        clean_up_pipeline: ActorHandle<CleanUpProcessor>,
        wallet_context: Arc<RwLock<WalletContext>>,
    ) -> Self {
        Self {
            store,
            broadcaster,
            encoder_keypair,
            clean_up_pipeline,
            wallet_context,
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

    async fn submit_winner_transaction(
        &self,
        report_vote: &ReportVote,
        agg: &EncoderAggregateSignature,
        evaluators: Vec<EncoderPublicKey>,
    ) -> ShardResult<()> {
        if report_vote.signed_report().winning_submission().encoder() != &self.encoder_keypair.public() {
            return Ok(());
        }
        
        let wallet_context = self.wallet_context.read().await;
        
        let (sender_address, gas_objects) = wallet_context
            .get_one_account()
            .await
            .map_err(|e| {
                error!("Could not get account from wallet: {}", e);
                ShardError::WalletError(e.to_string())
            })?;
        
        let tx_data = TransactionData::new(
            TransactionKind::ReportWinner {
                shard_auth_token: bcs::to_bytes(report_vote.auth_token())
                    .map_err(|e| ShardError::SerializationError(e.to_string()))?,
                shard_input_ref: report_vote.auth_token().shard_input_ref(),
                signed_report: bcs::to_bytes(&report_vote.signed_report())
                    .map_err(|e| ShardError::SerializationError(e.to_string()))?,
                signature: bcs::to_bytes(agg)
                    .map_err(|e| ShardError::SerializationError(e.to_string()))?,
                signers: evaluators,
            },
            sender_address,
            gas_objects,
        );
        
        let signed_tx = wallet_context.sign_transaction(&tx_data).await;
        
        wallet_context
            .execute_transaction_may_fail(signed_tx)
            .await
            .map_err(|e| {
                error!("Report winner tx failed: {}", e);
                ShardError::TransactionFailed(e.to_string())
            })?;
        
        Ok(())
    }
}

#[async_trait]
impl<E: EncoderInternalNetworkClient> Processor for ReportVoteProcessor<E> {
    type Input = (Shard, Verified<ReportVote>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (shard, report_vote) = msg.input;

            let _ = self.store.add_shard_stage_message(
                &shard,
                ShardStage::ReportVote,
                report_vote.author(),
            )?;
            self.store.add_report_vote(&shard, &report_vote)?;
            info!(
                "Starting track_valid_scores for scorer: {:?}",
                report_vote.author()
            );

            let all_scores = self.store.get_all_report_votes(&shard)?;
            debug!(
                "Current score count: {}, quorum_threshold: {}",
                all_scores.len(),
                shard.quorum_threshold()
            );

            debug!("{:?}", all_scores);

            let matching_report_votes: Vec<ReportVote> = all_scores
                .iter()
                .filter(|sv| {
                    report_vote.signed_report().deref() == sv.signed_report().deref()
                })
                .cloned()
                .collect();

            info!(
                "Found matching scores: {}, quorum_threshold: {}",
                matching_report_votes.len(),
                shard.quorum_threshold()
            );


            if matching_report_votes.len() >= shard.quorum_threshold() as usize {
                let _ = self.store.add_shard_stage_dispatch(&shard, ShardStage::Finalize)?;
                info!("QUORUM OF MATCHING SCORES - Aggregating signatures");

                let (signatures, evaluators): (Vec<EncoderSignature>, Vec<EncoderPublicKey>) = {
                    let mut sigs = Vec::new();
                    let mut evaluators = Vec::new();
                    for report_vote in matching_report_votes.iter() {
                        let sig = EncoderSignature::from_bytes(
                            &report_vote.signed_report().raw_signature(),
                        )
                        .map_err(ShardError::SignatureAggregationFailure)?;
                        sigs.push(sig);
                        evaluators.push(report_vote.author().clone());
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
                    .add_aggregate_score(&shard, (agg.clone(), evaluators.clone()))?;

                self.submit_winner_transaction(&report_vote, &agg, evaluators).await?;
                

                info!(
                    "SHARD CONSENSUS COMPLETE - Aggregate score stored: {:?}",
                    agg
                );

                self.clean_up_pipeline
                    .process(shard.clone(), msg.cancellation.clone())
                    .await?;
            } else {
                debug!(
                    "Not enough matching scores yet - waiting for more scores. Matching scores: {}, \
                    quorum_threshold: {}",
                    matching_report_votes.len(),
                    shard.quorum_threshold()
                );
            }

            info!("Completed track_valid_scores");
            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}
