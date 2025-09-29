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
use objects::{
    networking::{downloader::Downloader, ObjectNetworkClient},
    storage::ObjectStorage,
};
use tokio_util::sync::CancellationToken;
use tracing::debug;
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    evaluation::{
        EvaluationInput, EvaluationInputV1, EvaluationOutputAPI, ProbeSetAPI, ProbeWeightAPI,
        ScoreAPI,
    },
    metadata::{DownloadableMetadata, DownloadableMetadataV1, Metadata},
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

pub(crate) struct EvaluationProcessor<
    O: ObjectNetworkClient,
    E: EncoderInternalNetworkClient,
    S: ObjectStorage,
    P: EvaluationClient,
> {
    store: Arc<dyn Store>,
    downloader: ActorHandle<Downloader<O, S>>,
    broadcaster: Arc<Broadcaster<E>>,
    encoder_keypair: Arc<EncoderKeyPair>,
    storage: Arc<S>,
    report_vote_pipeline: ActorHandle<ReportVoteProcessor<E>>,
    evaluation_client: Arc<P>,
    context: Context,
}

impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > EvaluationProcessor<O, E, S, P>
{
    pub(crate) fn new(
        store: Arc<dyn Store>,
        downloader: ActorHandle<Downloader<O, S>>,
        broadcaster: Arc<Broadcaster<E>>,
        encoder_keypair: Arc<EncoderKeyPair>,
        storage: Arc<S>,
        report_vote_pipeline: ActorHandle<ReportVoteProcessor<E>>,
        evaluation_client: Arc<P>,
        context: Context,
    ) -> Self {
        Self {
            store,
            downloader,
            broadcaster,
            encoder_keypair,
            storage,
            report_vote_pipeline,
            evaluation_client,
            context,
        }
    }
}

#[async_trait]
impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > Processor for EvaluationProcessor<O, E, S, P>
{
    type Input = (ShardAuthToken, Shard);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let (auth_token, shard) = msg.input;

            let all_submissions = self.store.get_all_submissions(&shard)?;
            let all_accepted_commits = self.store.get_all_accepted_commits(&shard)?;

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
                        .map(|_| submission)
                })
                .collect();

            // TODO: ANY ACCEPTED COMMITS THAT DO NOT REVEAL SHOULD BE TALLIED

            valid_submissions.sort_by(|a, b| {
                a.score()
                    .value()
                    .partial_cmp(&b.score().value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_score = self
                .process_submissions(
                    valid_submissions,
                    auth_token.metadata_commitment().metadata(),
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

impl<
        O: ObjectNetworkClient,
        E: EncoderInternalNetworkClient,
        S: ObjectStorage,
        P: EvaluationClient,
    > EvaluationProcessor<O, E, S, P>
{
    async fn process_submissions(
        &self,
        submissions: Vec<Submission>,
        data_metadata: Metadata,
        context: &Context,
        cancellation: CancellationToken,
    ) -> ShardResult<Submission> {
        for submission in submissions {
            let result: ShardResult<()> = {
                if submission.encoder().inner() == self.context.own_encoder_key().inner() {
                    // skip early if your own representations
                    return Ok(submission);
                }
                let (peer, address) = context
                    .object_server(submission.encoder())
                    .ok_or(ShardError::MissingData)?;
                // TODO: actually store things in object storage
                if !cfg!(msim) {
                    let downloadable_metadata =
                        DownloadableMetadata::V1(DownloadableMetadataV1::new(
                            peer.clone(),
                            address.clone(),
                            submission.metadata().clone(),
                        ));
                    self.downloader
                        .process(downloadable_metadata, cancellation.clone())
                        .await?;

                    for probe in submission.probe_set().probe_weights() {
                        let (peer, address) = context
                            .object_server(probe.encoder())
                            .ok_or(ShardError::MissingData)?;
                        let downloadable_metadata = DownloadableMetadata::V1(
                            DownloadableMetadataV1::new(peer, address, probe.metadata().clone()),
                        );
                        self.downloader
                            .process(downloadable_metadata, cancellation.clone())
                            .await?;
                    }
                    // TODO: IF DOWNLOADING FAILS, TALLY
                }

                let evaluation_input = EvaluationInput::V1(EvaluationInputV1::new(
                    data_metadata.clone(),
                    submission.metadata().clone(),
                    submission.probe_set().clone(),
                    peer,
                    address,
                ));
                let evaluation_timeout = Duration::from_secs(1);

                // pass into the evaluation step
                let evaluation_output = self
                    .evaluation_client
                    .evaluation(evaluation_input, evaluation_timeout)
                    .await
                    .map_err(ShardError::EvaluationError)?;

                // TODO: this verification should be handled very differently allowing for an epsilon of error due to differences in
                if true {
                    // TODO: IF VERIFICATION FAILS, TALLY
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
