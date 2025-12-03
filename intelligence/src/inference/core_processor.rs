use std::{marker::PhantomData, sync::Arc};

use async_trait::async_trait;
use object_store::ObjectStore;
use objects::{
    downloader::ObjectDownloader,
    readers::{store::ObjectStoreReader, url::ObjectHttpClient},
    stores::{EphemeralStore, PersistentStore},
};
use tokio_util::sync::CancellationToken;
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
    metadata::{DownloadMetadata, Metadata, ObjectPath},
};

use crate::inference::{
    module::{ModuleClient, ModuleInput, ModuleInputV1, ModuleOutputAPI},
    InferenceInput, InferenceInputAPI, InferenceOutput, InferenceOutputV1,
};

pub struct InferenceCoreProcessor<
    PS: ObjectStore,
    ES: ObjectStore,
    P: PersistentStore<PS>,
    E: EphemeralStore<ES>,
    M: ModuleClient,
> {
    persistent_store: P,
    ephemeral_store: E,
    module_client: Arc<M>,
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
        M: ModuleClient,
    > InferenceCoreProcessor<PS, ES, P, E, M>
{
    pub fn new(
        persistent_store: P,
        ephemeral_store: E,
        module_client: Arc<M>,
        downloader: Arc<ObjectDownloader>,
        object_http_client: ObjectHttpClient,
    ) -> Self {
        Self {
            persistent_store,
            ephemeral_store,
            module_client,
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

    async fn store_to_persistent(
        &self,
        object_path: &ObjectPath,
        metadata: &Metadata,
    ) -> ShardResult<()> {
        let reader = Arc::new(ObjectStoreReader::new(
            self.ephemeral_store.object_store().clone(),
            object_path.clone(),
        ));
        self.downloader
            .download(
                reader,
                self.persistent_store.object_store().clone(),
                object_path.clone(),
                metadata.clone(),
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
        M: ModuleClient,
    > Processor for InferenceCoreProcessor<PS, ES, P, E, M>
{
    type Input = InferenceInput;
    type Output = InferenceOutput;

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<Self::Output> = async {
            let input = msg.input;
            self.load_data(
                input.object_path(),
                input.download_metadata(),
                msg.cancellation.clone(),
            )
            .await?;
            let module_input = ModuleInput::V1(ModuleInputV1::new(
                input.epoch(),
                input.download_metadata().metadata().clone(),
                input.object_path().clone(),
            ));

            let module_output = self
                .module_client
                .call(module_input, self.ephemeral_store.object_store().clone())
                .await
                .map_err(ShardError::InferenceError)?;

            self.store_to_persistent(input.object_path(), input.download_metadata().metadata())
                .await?;

            let download_metadata = self
                .persistent_store
                .download_metadata(
                    module_output.object_path().clone(),
                    module_output.metadata().clone(),
                )
                .await
                .map_err(ShardError::ObjectError)?;

            self.store_to_persistent(module_output.object_path(), download_metadata.metadata())
                .await?;

            Ok(InferenceOutput::V1(InferenceOutputV1::new(
                download_metadata,
                module_output.probe_set().clone(),
            )))
        }
        .await;
        let _ = msg.sender.send(result);
    }
    fn shutdown(&mut self) {}
}
