use std::{marker::PhantomData, sync::Arc};

use async_trait::async_trait;
use object_store::ObjectStore;
use objects::{
    downloader::ObjectDownloader,
    readers::{
        store::ObjectStoreReader,
        url::{ObjectHttpClient, ObjectHttpReader},
    },
    stores::{EphemeralStore, PersistentStore},
};
use tokio_util::sync::CancellationToken;

use types::{
    actors::{ActorMessage, Processor},
    error::{ShardError, ShardResult},
    evaluation::{EvaluationInput, EvaluationInputAPI, EvaluationOutput, EvaluationOutputV1},
    metadata::{DownloadMetadata, ObjectPath},
};

use crate::evaluation::{EvaluatorClient, EvaluatorInput, EvaluatorInputV1, EvaluatorOutputAPI};

pub struct EvaluationCoreProcessor<
    PS: ObjectStore,
    ES: ObjectStore,
    P: PersistentStore<PS>,
    E: EphemeralStore<ES>,
    C: EvaluatorClient,
> {
    persistent_store: P,
    ephemeral_store: E,
    evaluator_client: Arc<C>,
    downloader: Arc<ObjectDownloader>,
    object_http_client: ObjectHttpClient,
    p_marker: PhantomData<PS>,
    e_marker: PhantomData<ES>,
}

impl<
        PS: ObjectStore,
        ES: ObjectStore,
        P: PersistentStore<PS>,
        E: EphemeralStore<ES>,
        C: EvaluatorClient,
    > EvaluationCoreProcessor<PS, ES, P, E, C>
{
    pub fn new(
        persistent_store: P,
        ephemeral_store: E,
        evaluator_client: Arc<C>,
        downloader: Arc<ObjectDownloader>,
        object_http_client: ObjectHttpClient,
    ) -> Self {
        Self {
            persistent_store,
            ephemeral_store,
            evaluator_client,
            downloader,
            object_http_client,
            p_marker: PhantomData,
            e_marker: PhantomData,
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
            let reader = Arc::new(ObjectStoreReader::new(
                self.persistent_store.object_store().clone(),
                object_path.clone(),
            ));
            self.downloader
                .download(
                    reader,
                    self.ephemeral_store.object_store().clone(),
                    object_path.clone(),
                    download_metadata.metadata().clone(),
                )
                .await
                .map_err(ShardError::ObjectError)?;
            return Ok(());
        }

        // In msim tests, data may be at a different path (e.g., uploads/xxx instead of inputs/xxx).
        // Check if data exists at the URL path in the store and copy to expected location.
        #[cfg(msim)]
        {
            use object_store::path::Path as ObjPath;
            use types::error::ObjectError;

            let url_path = ObjPath::from(download_metadata.url().path().trim_start_matches('/'));
            if let Ok(result) = self.persistent_store.object_store().get(&url_path).await {
                if let Ok(bytes) = result.bytes().await {
                    self.ephemeral_store
                        .object_store()
                        .put(&object_path.path(), bytes.into())
                        .await
                        .map_err(|e| ShardError::ObjectError(ObjectError::ObjectStoreError(e)))?;
                    return Ok(());
                }
            }
        }

        let reader = Arc::new(
            self.object_http_client
                .get_reader(download_metadata)
                .await
                .map_err(ShardError::ObjectError)?,
        );

        self.downloader
            .download(
                reader,
                self.ephemeral_store.object_store().clone(),
                object_path.clone(),
                download_metadata.metadata().clone(),
            )
            .await
            .map_err(ShardError::ObjectError)?;

        Ok(())
    }
}

#[async_trait]
impl<
        PS: ObjectStore,
        ES: ObjectStore,
        P: PersistentStore<PS>,
        E: EphemeralStore<ES>,
        C: EvaluatorClient,
    > Processor for EvaluationCoreProcessor<PS, ES, P, E, C>
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

            self.load_data(
                input.probe_object_path(),
                input.probe_download_metadata(),
                msg.cancellation.clone(),
            )
            .await?;

            let evaluator_input = EvaluatorInput::V1(EvaluatorInputV1::new(
                input.input_object_path().clone(),
                input.embedding_object_path().clone(),
                input.probe_object_path().clone(),
            ));

            let evaluator_output = self
                .evaluator_client
                .call(evaluator_input, self.ephemeral_store.object_store().clone())
                .await
                .map_err(ShardError::EvaluationError)?;

            let reader = Arc::new(ObjectStoreReader::new(
                self.ephemeral_store.object_store().clone(),
                input.input_object_path().clone(),
            ));
            self.downloader
                .download(
                    reader,
                    self.persistent_store.object_store().clone(),
                    input.input_object_path().clone(),
                    input.input_download_metadata().metadata().clone(),
                )
                .await
                .map_err(ShardError::ObjectError)?;

            let reader = Arc::new(ObjectStoreReader::new(
                self.ephemeral_store.object_store().clone(),
                input.embedding_object_path().clone(),
            ));
            self.downloader
                .download(
                    reader,
                    self.persistent_store.object_store().clone(),
                    input.embedding_object_path().clone(),
                    input.embedding_download_metadata().metadata().clone(),
                )
                .await
                .map_err(ShardError::ObjectError)?;

            let reader = Arc::new(ObjectStoreReader::new(
                self.ephemeral_store.object_store().clone(),
                input.probe_object_path().clone(),
            ));
            self.downloader
                .download(
                    reader,
                    self.persistent_store.object_store().clone(),
                    input.probe_object_path().clone(),
                    input.probe_download_metadata().metadata().clone(),
                )
                .await
                .map_err(ShardError::ObjectError)?;

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
