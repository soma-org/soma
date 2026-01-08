use std::{collections::HashMap, sync::Arc, time::Duration};

use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::{ShardStage, Store},
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        context::Context,
        report_vote::{ReportVote, ReportVoteV1},
    },
};
use async_trait::async_trait;
use fastcrypto::traits::KeyPair;
use intelligence::evaluation::networking::EvaluationClient;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    committee::Epoch,
    error::{ShardError, ShardResult},
    evaluation::{
        Embedding, EvaluationInput, EvaluationInputV1, EvaluationScoresAPI as _,
        TargetDetailsAPI as _,
    },
    metadata::{DownloadMetadata, MetadataAPI, ObjectPath},
    report::{Report, ReportV1},
    shard::{InputAPI, Shard},
    shard_crypto::{
        digest::Digest,
        keys::{EncoderKeyPair, EncoderPublicKey},
        scope::Scope,
        signed::Signed,
        verified::Verified,
    },
    submission::SubmissionAPI,
};
use types::{shard::ShardAuthToken, submission::Submission};

use super::report_vote::ReportVoteProcessor;

pub(crate) struct EvaluationProcessor<C: EncoderInternalNetworkClient, E: EvaluationClient> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<C>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    report_vote_pipeline: ActorHandle<ReportVoteProcessor<C>>,
    evaluation_client: Arc<E>,
    context: Context,
}

impl<C: EncoderInternalNetworkClient, E: EvaluationClient> EvaluationProcessor<C, E> {
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<C>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        report_vote_pipeline: ActorHandle<ReportVoteProcessor<C>>,
        evaluation_client: Arc<E>,
        context: Context,
    ) -> Self {
        Self {
            store,
            broadcaster,
            encoder_keypair,
            report_vote_pipeline,
            evaluation_client,
            context,
        }
    }
}

#[async_trait]
impl<C: EncoderInternalNetworkClient, E: EvaluationClient> Processor for EvaluationProcessor<C, E> {
    type Input = (ShardAuthToken, Shard);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (auth_token, shard) = msg.input;

            let shard_digest = shard.digest()?;

            let all_submissions = self.store.get_all_submissions(&shard)?;
            let all_accepted_commits = self.store.get_all_accepted_commits(&shard)?;
            let input = self.store.get_input(&shard)?;
            let input_download_metadata = input.input_download_metadata();

            let accepted_lookup: HashMap<EncoderPublicKey, Digest<Submission>> =
                all_accepted_commits
                    .clone()
                    .into_iter()
                    .map(|(encoder, digest)| (encoder, digest))
                    .collect();

            let mut valid_submissions: Vec<Submission> = all_submissions
                .into_iter()
                .filter_map(|(submission, _instant)| {
                    accepted_lookup
                        .get(submission.encoder())
                        .filter(|accepted_digest| {
                            **accepted_digest == Digest::new(&submission).unwrap()
                        })
                        .map(|_| (submission))
                })
                .collect();

            // TODO: ANY ACCEPTED COMMITS THAT DO NOT REVEAL SHOULD BE TALLIED

            valid_submissions.sort_by(|a, b| {
                a.evaluation_scores()
                    .score()
                    .partial_cmp(&b.evaluation_scores().score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_submission = self
                .process_submissions(
                    auth_token.epoch(),
                    shard_digest.clone(),
                    input_download_metadata.clone(),
                    valid_submissions,
                    input.target_embedding().map(Into::into),
                    msg.cancellation.clone(),
                )
                .await?;

            let target_scores = best_submission
                .target_details()
                .as_ref()
                .map(|td| td.target_scores().clone());

            let report = Report::V1(ReportV1::new(
                best_submission.encoder().clone(),
                shard_digest,
                input_download_metadata.clone(),
                best_submission.embedding_download_metadata().clone(),
                best_submission.probe_encoder().clone(),
                best_submission.evaluation_scores().clone(),
                best_submission.summary_embedding().clone(),
                best_submission.sampled_embedding().clone(),
                target_scores,
            ));

            let inner_keypair = self.encoder_keypair.inner().copy();

            let signed_report =
                Signed::new(report, Scope::ShardReport, &inner_keypair.copy().private()).unwrap();

            let report_vote = ReportVote::V1(ReportVoteV1::new(
                auth_token,
                self.encoder_keypair.public(),
                signed_report,
            ));

            let verified = Verified::from_trusted(report_vote).unwrap();
            let _ = self
                .store
                .add_shard_stage_dispatch(&shard, ShardStage::ReportVote)?;

            self.report_vote_pipeline
                .process((shard.clone(), verified.clone()), msg.cancellation.clone())
                .await?;

            // Broadcast to other encoders
            self.broadcaster
                .broadcast(
                    verified.clone(),
                    shard.encoders(),
                    |client, peer, verified_type| async move {
                        client
                            .send_report_vote(&peer, &verified_type, MESSAGE_TIMEOUT)
                            .await?;
                        Ok(())
                    },
                )
                .await?;
            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {}
}

impl<C: EncoderInternalNetworkClient, E: EvaluationClient> EvaluationProcessor<C, E> {
    async fn process_submissions(
        &self,
        epoch: Epoch,
        shard_digest: Digest<Shard>,
        input_download_metadata: DownloadMetadata,
        submissions: Vec<Submission>,
        target_embedding: Option<Embedding>,
        cancellation: CancellationToken,
    ) -> ShardResult<Submission> {
        for submission in submissions {
            let result: ShardResult<()> = {
                if submission.encoder().inner() == self.context.own_encoder_key().inner() {
                    // skip early if your own representations
                    return Ok(submission);
                }

                let embedding_download_metadata = submission.embedding_download_metadata();
                let probe_download_metadata =
                    self.context.probe(epoch, submission.probe_encoder())?;

                let evaluation_input = EvaluationInput::V1(EvaluationInputV1::new(
                    input_download_metadata.clone(),
                    ObjectPath::Inputs(
                        epoch,
                        shard_digest.clone(),
                        input_download_metadata.metadata().checksum(),
                    ),
                    embedding_download_metadata.clone(),
                    ObjectPath::Embeddings(
                        epoch,
                        shard_digest.clone(),
                        embedding_download_metadata.metadata().checksum(),
                    ),
                    submission.probe_encoder().clone(),
                    probe_download_metadata.clone(),
                    ObjectPath::Probes(epoch, probe_download_metadata.metadata().checksum()),
                    target_embedding.clone(),
                ));

                let evaluation_timeout = Duration::from_secs(1);

                // pass into the evaluation step
                let evaluation_output = self
                    .evaluation_client
                    .evaluation(evaluation_input, evaluation_timeout)
                    .await
                    .map_err(ShardError::EvaluationError)?;
                // TODO: check summary digest

                // TODO: this verification should be handled very differently allowing for an epsilon of error due to differences in
                if true {
                    // TODO: IF VERIFICATION FAILTALLY
                    // floating point math on various accelerators
                    return Ok(submission); // Short-circuit and return the first valid reveal
                } else {
                    Err(ShardError::FailedTypeVerification(
                        "reveal scores did not match".to_string(),
                    ))
                }
            };

            match result {
                Ok(_) => return Ok(submission),
                Err(_) => continue,
            }
        }

        // Return an error if no reveal passes verification
        Err(ShardError::ShardFailure(
            "no reveals were valid".to_string(),
        ))
    }
}
