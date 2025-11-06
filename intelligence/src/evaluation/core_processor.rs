use std::sync::Arc;

use async_trait::async_trait;
use object_store::ObjectStore;
use tokio_util::sync::CancellationToken;

use objects::{
    networking::{downloader::Downloader, transfer::Transfer},
    EphemeralStore, PersistentStore,
};
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    evaluation::{
        EvaluationInput, EvaluationInputAPI, EvaluationOutput, EvaluationOutputV1, ProbeSetAPI,
        ProbeWeightAPI,
    },
    metadata::{DownloadMetadata, ObjectPath},
};

use crate::evaluation::{EvaluatorClient, EvaluatorInput, EvaluatorInputV1, EvaluatorOutputAPI};

pub struct EvaluationCoreProcessor<P: PersistentStore, E: EphemeralStore, C: EvaluatorClient> {
    persistent_store: P,
    ephemeral_store: E,
    evaluator_client: Arc<C>,
    downloader: ActorHandle<Downloader>,
}

impl<P: PersistentStore, E: EphemeralStore, C: EvaluatorClient> EvaluationCoreProcessor<P, E, C> {
    pub fn new(
        persistent_store: P,
        ephemeral_store: E,
        evaluator_client: Arc<C>,
        downloader: ActorHandle<Downloader>,
    ) -> Self {
        Self {
            persistent_store,
            ephemeral_store,
            evaluator_client,
            downloader,
        }
    }
    async fn load_data(
        &self,
        object_path: &ObjectPath,
        download_metadata: &DownloadMetadata,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        if self
            .ephemeral_store
            .object_store()
            .head(&object_path.path())
            .await
            .is_ok()
        {
            return Ok(());
        }

        // Check persistent.
        if self
            .persistent_store
            .object_store()
            .head(&object_path.path())
            .await
            .is_ok()
        {
            Transfer::transfer(
                self.persistent_store.object_store().clone(),
                self.ephemeral_store.object_store().clone(),
                object_path,
            )
            .await
            .map_err(ShardError::ObjectError)?;
            return Ok(());
        }

        // Download from remote.
        self.downloader
            .process(
                (
                    download_metadata.clone(),
                    object_path.clone(),
                    self.ephemeral_store.object_store().clone(),
                ),
                cancellation,
            )
            .await?;

        Ok(())
    }
}

#[async_trait]
impl<P: PersistentStore, E: EphemeralStore, C: EvaluatorClient> Processor
    for EvaluationCoreProcessor<P, E, C>
{
    type Input = EvaluationInput;
    type Output = EvaluationOutput;

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<Self::Output> = async {
            let input = msg.input;

            self.load_data(
                input.input_object_path(),
                input.input_download_metadata(),
                msg.cancellation.clone(),
            )
            .await?;
            self.load_data(
                input.embedding_object_path(),
                input.embedding_download_metadata(),
                msg.cancellation.clone(),
            )
            .await?;

            for (_encoder, (download_metadata, object_path)) in input.probe_set_data() {
                self.load_data(object_path, download_metadata, msg.cancellation.clone())
                    .await?;
            }

            let mut probes = Vec::new();
            for pw in input.probe_set().probe_weights() {
                if let Some((_, object_path)) = input.probe_set_data().get(pw.encoder()) {
                    probes.push((pw.weight(), object_path.clone()));
                } else {
                    return Err(ShardError::MissingData);
                }
            }

            let evaluator_input = EvaluatorInput::V1(EvaluatorInputV1::new(
                input.input_object_path().clone(),
                input.embedding_object_path().clone(),
                probes,
            ));

            let evaluator_output = self
                .evaluator_client
                .call(evaluator_input, self.ephemeral_store.object_store().clone())
                .await
                .map_err(ShardError::EvaluationError)?;

            Transfer::transfer(
                self.ephemeral_store.object_store().clone(),
                self.persistent_store.object_store().clone(),
                &input.input_object_path(),
            )
            .await
            .map_err(ShardError::ObjectError)?;

            Transfer::transfer(
                self.ephemeral_store.object_store().clone(),
                self.persistent_store.object_store().clone(),
                &input.embedding_object_path(),
            )
            .await
            .map_err(ShardError::ObjectError)?;

            for (_encoder, (_download_metadata, object_path)) in input.probe_set_data() {
                Transfer::transfer(
                    self.ephemeral_store.object_store().clone(),
                    self.persistent_store.object_store().clone(),
                    &object_path,
                )
                .await
                .map_err(ShardError::ObjectError)?;
            }

            Ok(EvaluationOutput::V1(EvaluationOutputV1::new(
                evaluator_output.score().clone(),
                evaluator_output.summary_digest().clone(),
            )))
        }
        .await;
        let _ = msg.sender.send(result);
    }
    fn shutdown(&mut self) {}
}
