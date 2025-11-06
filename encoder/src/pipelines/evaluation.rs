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
use intelligence::evaluation::messaging::EvaluationClient;
use object_store::ObjectStore;
use objects::networking::downloader::Downloader;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    committee::Epoch,
    error::{ShardError, ShardResult},
    evaluation::{EvaluationInput, EvaluationInputV1, ProbeSetAPI, ProbeWeightAPI, ScoreAPI},
    metadata::{DownloadMetadata, Metadata, MetadataAPI, ObjectPath},
    report::{Report, ReportV1},
    shard::Shard,
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
            let input_download_metadata = self.store.get_input_download_metadata(&shard)?;

            let accepted_lookup: HashMap<EncoderPublicKey, Digest<Submission>> =
                all_accepted_commits
                    .clone()
                    .into_iter()
                    .map(|(encoder, digest)| (encoder, digest))
                    .collect();

            let mut valid_submissions: Vec<(Submission, DownloadMetadata)> = all_submissions
                .into_iter()
                .filter_map(|(submission, _instant, embedding_download_metadata)| {
                    accepted_lookup
                        .get(submission.encoder())
                        .filter(|accepted_digest| {
                            **accepted_digest == Digest::new(&submission).unwrap()
                        })
                        .map(|_| (submission, embedding_download_metadata))
                })
                .collect();

            // TODO: ANY ACCEPTED COMMITS THAT DO NOT REVEAL SHOULD BE TALLIED

            valid_submissions.sort_by(|a, b| {
                a.0.score()
                    .value()
                    .partial_cmp(&b.0.score().value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_score = self
                .process_submissions(
                    auth_token.epoch(),
                    shard_digest.clone(),
                    input_download_metadata,
                    valid_submissions,
                    auth_token.metadata(),
                    &self.context,
                    msg.cancellation.clone(),
                )
                .await?;

            debug!("BEST SCORE: {:?}", best_score);

            let report = Report::V1(ReportV1::new(best_score, all_accepted_commits));

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
        submissions: Vec<(Submission, DownloadMetadata)>,
        data_metadata: Metadata,
        context: &Context,
        cancellation: CancellationToken,
    ) -> ShardResult<Submission> {
        for (submission, embedding_download_metadata) in submissions {
            let result: ShardResult<()> = {
                if submission.encoder().inner() == self.context.own_encoder_key().inner() {
                    // skip early if your own representations
                    return Ok(submission);
                }
                let mut probe_set_download_metadata = HashMap::new();
                for pw in submission.probe_set().probe_weights() {
                    let probe_download_metadata = self.context.probe(epoch, pw.encoder())?;
                    let probe_object_path =
                        ObjectPath::Probes(epoch, probe_download_metadata.metadata().checksum());
                    probe_set_download_metadata.insert(
                        pw.encoder().clone(),
                        (probe_download_metadata, probe_object_path),
                    );
                }

                let evaluation_input = EvaluationInput::V1(EvaluationInputV1::new(
                    input_download_metadata.clone(),
                    ObjectPath::Inputs(
                        epoch,
                        shard_digest,
                        input_download_metadata.metadata().checksum(),
                    ),
                    embedding_download_metadata.clone(),
                    ObjectPath::Embeddings(
                        epoch,
                        shard_digest,
                        embedding_download_metadata.metadata().checksum(),
                    ),
                    probe_set_download_metadata,
                    submission.probe_set().clone(),
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
