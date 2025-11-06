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
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tracing::error;
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    evaluation::{
        EvaluationInput, EvaluationInputV1, EvaluationOutputAPI, ProbeSetAPI, ProbeWeightAPI,
    },
    metadata::{
        DownloadMetadata, MetadataAPI, MtlsDownloadMetadata, MtlsDownloadMetadataV1, ObjectPath,
    },
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        keys::{EncoderKeyPair, EncoderPublicKey},
        verified::Verified,
    },
    submission::{Submission, SubmissionV1},
};
use types::{
    multiaddr::Multiaddr,
    shard::{Input, InputAPI},
};
use url::Url;

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

            let input_download_metadata = verified_input.download_metadata();
            let input_object_path = ObjectPath::Inputs(
                epoch,
                shard_digest,
                input_download_metadata.metadata().checksum(),
            );

            self.store
                .add_input_download_metadata(&shard, input_download_metadata.clone())?;

            let inference_input = InferenceInput::V1(InferenceInputV1::new(
                epoch,
                input_object_path.clone(),
                input_download_metadata.clone(),
            ));

            // TODO: make this adjusted with size and coefficient configured by Parameters
            let inference_timeout = Duration::from_secs(60);

            let inference_output = self
                .inference_client
                .inference(inference_input, inference_timeout)
                .await
                .map_err(ShardError::InferenceError)?;

            let mut probe_set_download_metadata = HashMap::new();
            for pw in inference_output.probe_set().probe_weights() {
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
                input_object_path,
                inference_output.download_metadata().clone(),
                ObjectPath::Embeddings(
                    epoch,
                    shard_digest,
                    inference_output.download_metadata().metadata().checksum(),
                ),
                probe_set_download_metadata,
                inference_output.probe_set().clone(),
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
                inference_output.download_metadata().metadata().clone(),
                inference_output.probe_set().clone(),
                evaluation_output.score(),
                evaluation_output.summary_digest().clone(),
            ));

            let submission_digest = Digest::new(&submission).map_err(ShardError::DigestFailure)?;
            let _ = self.store.add_submission(
                &shard,
                submission,
                inference_output.download_metadata().clone(),
            )?;

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
