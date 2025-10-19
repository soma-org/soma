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
use intelligence::evaluation::messaging::EvaluationClient;

use fastcrypto::traits::KeyPair;
use intelligence::inference::{
    InferenceClient, InferenceInput, InferenceInputV1, InferenceOutputAPI,
};
use objects::{
    networking::{downloader::Downloader, external_service::ExternalClientPool},
    storage::ObjectStorage,
};
use std::{sync::Arc, time::Duration};
use tracing::error;
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    evaluation::{EvaluationInput, EvaluationInputV1, EvaluationOutputAPI},
    metadata::{DownloadableMetadata, DownloadableMetadataV1, Metadata, MetadataAPI, SignedParams},
    shard::Shard,
    shard_crypto::{digest::Digest, keys::EncoderKeyPair, verified::Verified},
    submission::{Submission, SubmissionV1},
};
use types::{
    multiaddr::Multiaddr,
    shard::{Input, InputAPI},
};

use super::commit::CommitProcessor;

pub(crate) struct InputProcessor<
    C: EncoderInternalNetworkClient,
    S: ObjectStorage,
    E: EvaluationClient,
    I: InferenceClient,
> {
    store: Arc<dyn Store>,
    downloader: ActorHandle<Downloader<ExternalClientPool, S>>,
    broadcaster: Arc<Broadcaster<C>>,
    inference_client: Arc<I>,
    evaluation_client: Arc<E>,
    encoder_keypair: Arc<EncoderKeyPair>,
    storage: Arc<S>,
    commit_pipeline: ActorHandle<CommitProcessor<C, S, E>>,
    local_object_server: Multiaddr,
    context: Context,
}

impl<
        C: EncoderInternalNetworkClient,
        S: ObjectStorage,
        E: EvaluationClient,
        I: InferenceClient,
    > InputProcessor<C, S, E, I>
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        store: Arc<dyn Store>,
        downloader: ActorHandle<Downloader<ExternalClientPool, S>>,
        broadcaster: Arc<Broadcaster<C>>,
        inference_client: Arc<I>,
        evaluation_client: Arc<E>,
        encoder_keypair: Arc<EncoderKeyPair>,
        storage: Arc<S>,
        commit_pipeline: ActorHandle<CommitProcessor<C, S, E>>,
        local_object_server: Multiaddr,
        context: Context,
    ) -> Self {
        Self {
            store,
            downloader,
            broadcaster,
            inference_client,
            evaluation_client,
            encoder_keypair,
            storage,
            commit_pipeline,
            local_object_server,
            context,
        }
    }
}

#[async_trait]
impl<
        C: EncoderInternalNetworkClient,
        S: ObjectStorage,
        E: EvaluationClient,
        I: InferenceClient,
    > Processor for InputProcessor<C, S, E, I>
{
    type Input = (Shard, Verified<Input>);
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let keypair = self.encoder_keypair.inner().copy();
        let result: ShardResult<()> = async {
            let (shard, verified_input) = msg.input;

            // the add_external_stage function will fail if the encoder has already dispatched to input processing
            // this stops redundant or conflicting messages from reaching the pipelines even on encoder restart
            let _ = self
                .store
                .add_shard_stage_dispatch(&shard, ShardStage::Input)?;
            let shard_digest = shard.digest()?;
            let metadata = verified_input.auth_token().metadata_commitment().metadata();

            if !cfg!(msim) {
                self.downloader
                    .process(
                        verified_input.downloadable_metadata().clone(),
                        msg.cancellation.clone(),
                    )
                    .await?;
            }

            let inference_input =
                InferenceInput::V1(InferenceInputV1::new(shard.epoch(), metadata.clone()));

            // TODO: make this adjusted with size and coefficient configured by Parameters
            let inference_timeout = Duration::from_secs(1);
            let inference_output = self
                .inference_client
                .call(inference_input, inference_timeout)
                .await
                .map_err(ShardError::IntelligenceError)?;
            // TODO need to rename from tmp to the appropriate shard?

            let (peer, address) = self
                .context
                .object_server(&self.encoder_keypair.public())
                .ok_or(ShardError::MissingData)?;

            let evaluation_input = EvaluationInput::V1(EvaluationInputV1::new(
                metadata,
                inference_output.metadata(),
                inference_output.probe_set(),
                address,
            ));
            // TODO: make this adjusted with size and coefficient configured by Parameters
            let evaluation_timeout = Duration::from_secs(1);

            let evaluation_output = self
                .evaluation_client
                .evaluation(evaluation_input, evaluation_timeout)
                .await
                .map_err(ShardError::EvaluationError)?;

            // send input data object path to inference
            // inference returns: probe_set, object path to representations/byte-ranges
            // send input data, probe set, and representations to evaluation
            // evaluation returns: score and summary embedding bytes
            // create and sign a reveal message
            // store in datastore
            // use the digest of the reveal to create and sign a commit message

            let submission = Submission::V1(SubmissionV1::new(
                self.encoder_keypair.public(),
                shard_digest,
                inference_output.metadata(),
                inference_output.probe_set(),
                evaluation_output.score(),
                evaluation_output.embedding_digest(),
            ));

            let downloadable_metadata = DownloadableMetadata::V1(DownloadableMetadataV1::new(
                Some(self.context.own_network_keypair().public()),
                Some(SignedParams::new(
                    inference_output.metadata().path().path(),
                    Some(self.context.network_public_keys(shard.encoders())?),
                    Duration::from_secs(100000),
                    &self.context.own_network_keypair(),
                )),
                self.context.internal_object_service_address(),
                inference_output.metadata(),
            ));

            let submission_digest = Digest::new(&submission).map_err(ShardError::DigestFailure)?;
            let _ = self
                .store
                .add_submission(&shard, submission, downloadable_metadata)?;

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
