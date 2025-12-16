use crate::{
    core::internal_broadcaster::Broadcaster,
    datastore::{ShardStage, Store},
    messaging::{EncoderInternalNetworkClient, MESSAGE_TIMEOUT},
    types::{
        commit::{Commit, CommitV1},
        context::Context,
    },
};
use async_trait::async_trait;
use intelligence::{
    evaluation::messaging::EvaluationClient, inference::messaging::InferenceClient,
};

use fastcrypto::traits::KeyPair;
use intelligence::inference::{InferenceInput, InferenceInputV1, InferenceOutputAPI};
use std::{sync::Arc, time::Duration};
use tracing::error;
use types::shard::{Input, InputAPI};
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    evaluation::{EvaluationInput, EvaluationInputV1, EvaluationOutputAPI},
    metadata::{MetadataAPI, ObjectPath},
    shard::Shard,
    shard_crypto::{digest::Digest, keys::EncoderKeyPair, verified::Verified},
    submission::{Submission, SubmissionV1},
};

use super::commit::CommitProcessor;

pub(crate) struct InputProcessor<
    C: EncoderInternalNetworkClient,
    E: EvaluationClient,
    I: InferenceClient,
> {
    store: Arc<dyn Store>,
    broadcaster: Arc<Broadcaster<C>>,
    inference_client: Arc<I>,
    evaluation_client: Arc<E>,
    encoder_keypair: Arc<EncoderKeyPair>,
    commit_pipeline: ActorHandle<CommitProcessor<C, E>>,
    context: Context,
}

impl<C: EncoderInternalNetworkClient, E: EvaluationClient, I: InferenceClient>
    InputProcessor<C, E, I>
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        store: Arc<dyn Store>,
        broadcaster: Arc<Broadcaster<C>>,
        inference_client: Arc<I>,
        evaluation_client: Arc<E>,
        encoder_keypair: Arc<EncoderKeyPair>,
        commit_pipeline: ActorHandle<CommitProcessor<C, E>>,
        context: Context,
    ) -> Self {
        Self {
            store,
            broadcaster,
            inference_client,
            evaluation_client,
            encoder_keypair,
            commit_pipeline,
            context,
        }
    }
}

#[async_trait]
impl<C: EncoderInternalNetworkClient, E: EvaluationClient, I: InferenceClient> Processor
    for InputProcessor<C, E, I>
{
    type Input = (Shard, Verified<Input>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let keypair = self.encoder_keypair.inner().copy();
        let result: ShardResult<()> = async {
            let (shard, verified_input) = msg.input;
            let epoch = shard.epoch();

            // the add_external_stage function will fail if the encoder has already dispatched to input processing
            // this stops redundant or conflicting messages from reaching the pipelines even on encoder restart
            let _ = self
                .store
                .add_shard_stage_dispatch(&shard, ShardStage::Input)?;
            let shard_digest = shard.digest()?;

            let input_download_metadata = verified_input.input_download_metadata();
            let input_object_path = ObjectPath::Inputs(
                epoch,
                shard_digest,
                input_download_metadata.metadata().checksum(),
            );

            self.store.add_input(&shard, verified_input.clone())?;

            let inference_input = InferenceInput::V1(InferenceInputV1::new(
                epoch,
                shard_digest,
                input_download_metadata.clone(),
                input_object_path.clone(),
            ));

            // TODO: make this adjusted with size and coefficient configured by Parameters
            let inference_timeout = Duration::from_secs(60);

            let inference_output = self
                .inference_client
                .inference(inference_input, inference_timeout)
                .await
                .map_err(ShardError::InferenceError)?;

            let probe_download_metadata = self
                .context
                .probe(epoch, inference_output.probe_encoder())?;

            let probe_object_path =
                ObjectPath::Probes(epoch, probe_download_metadata.metadata().checksum());

            let evaluation_input = EvaluationInput::V1(EvaluationInputV1::new(
                input_download_metadata.clone(),
                input_object_path,
                inference_output.output_download_metadata().clone(),
                ObjectPath::Embeddings(
                    epoch,
                    shard_digest,
                    inference_output
                        .output_download_metadata()
                        .metadata()
                        .checksum(),
                ),
                inference_output.probe_encoder().clone(),
                probe_download_metadata,
                probe_object_path,
                verified_input.target_embedding().map(Into::into),
            ));

            // TODO: make this based on size
            let evaluation_timeout = Duration::from_secs(60);

            let evaluation_output = self
                .evaluation_client
                .evaluation(evaluation_input, evaluation_timeout)
                .await
                .map_err(ShardError::EvaluationError)?;

            let submission = Submission::V1(SubmissionV1::new(
                self.encoder_keypair.public(),
                shard_digest,
                input_download_metadata.clone(),
                inference_output.output_download_metadata().clone(),
                inference_output.probe_encoder().clone(),
                evaluation_output.evaluation_scores().clone(),
                evaluation_output.summary_embedding().clone(),
                evaluation_output.sampled_embedding().clone(),
                evaluation_output.target_details().clone(),
            ));

            let submission_digest = Digest::new(&submission).map_err(ShardError::DigestFailure)?;
            let _ = self.store.add_submission(&shard, submission)?;

            let commit = Commit::V1(CommitV1::new(
                verified_input.auth_token().clone(),
                self.encoder_keypair.public(),
                submission_digest,
            ));

            let verified_commit = Verified::from_trusted(commit).unwrap();

            let _ = self
                .store
                .add_shard_stage_dispatch(&shard, ShardStage::Commit)?;

            self.commit_pipeline
                .process(
                    (shard.clone(), verified_commit.clone()),
                    msg.cancellation.clone(),
                )
                .await?;

            // Broadcast to other encoders
            self.broadcaster
                .broadcast(
                    verified_commit.clone(),
                    shard.encoders(),
                    |client, peer, verified_type| async move {
                        client
                            .send_commit(&peer, &verified_type, MESSAGE_TIMEOUT)
                            .await?;
                        Ok(())
                    },
                )
                .await?;

            Ok(())
        }
        .await;
        if let Err(Err(err)) = msg.sender.send(result) {
            error!("Input Pipeline Error: {:?}", err);
        }
    }

    fn shutdown(&mut self) {}
}
