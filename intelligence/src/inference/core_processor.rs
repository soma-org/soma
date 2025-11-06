use std::sync::Arc;

use async_trait::async_trait;
use object_store::ObjectStore;
use objects::{
    networking::{downloader::Downloader, transfer::Transfer},
    EphemeralStore, PersistentStore,
};
use tokio_util::sync::CancellationToken;
use types::{
    actors::{ActorHandle, ActorMessage, Processor},
    error::{ShardError, ShardResult},
};

use crate::inference::{
    module::{ModuleClient, ModuleInput, ModuleInputV1, ModuleOutputAPI},
    InferenceInput, InferenceInputAPI, InferenceOutput, InferenceOutputV1,
};

pub struct InferenceCoreProcessor<P: PersistentStore, E: EphemeralStore, M: ModuleClient> {
    persistent_store: P,
    ephemeral_store: E,
    downloader: ActorHandle<Downloader>,
    module_client: Arc<M>,
}

impl<P: PersistentStore, E: EphemeralStore, M: ModuleClient> InferenceCoreProcessor<P, E, M> {
    pub fn new(
        persistent_store: P,
        ephemeral_store: E,
        downloader: ActorHandle<Downloader>,
        module_client: Arc<M>,
    ) -> Self {
        Self {
            persistent_store,
            ephemeral_store,
            downloader,
            module_client,
        }
    }

    async fn load_input_data(
        &self,
        input: &InferenceInput,
        cancellation: CancellationToken,
    ) -> ShardResult<()> {
        if self
            .ephemeral_store
            .object_store()
            .head(&input.object_path().path())
            .await
            .is_ok()
        {
            return Ok(());
        }

        // Check persistent.
        if self
            .persistent_store
            .object_store()
            .head(&input.object_path().path())
            .await
            .is_ok()
        {
            Transfer::transfer(
                self.persistent_store.object_store().clone(),
                self.ephemeral_store.object_store().clone(),
                &input.object_path(),
            )
            .await
            .map_err(ShardError::ObjectError)?;
            return Ok(());
        }

        // Download from remote.
        self.downloader
            .process(
                (
                    input.download_metadata().clone(),
                    input.object_path().clone(),
                    self.ephemeral_store.object_store().clone(),
                ),
                cancellation,
            )
            .await?;

        Ok(())
    }
}

#[async_trait]
impl<P: PersistentStore, E: EphemeralStore, M: ModuleClient> Processor
    for InferenceCoreProcessor<P, E, M>
{
    type Input = InferenceInput;
    type Output = InferenceOutput;

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<Self::Output> = async {
            let input = msg.input;
            self.load_input_data(&input, msg.cancellation.clone())
                .await?;
            let module_input = ModuleInput::V1(ModuleInputV1::new(
                input.epoch(),
                input.object_path().clone(),
            ));

            let module_output = self
                .module_client
                .call(module_input, self.ephemeral_store.object_store().clone())
                .await
                .map_err(ShardError::InferenceError)?;

            Transfer::transfer(
                self.ephemeral_store.object_store().clone(),
                self.persistent_store.object_store().clone(),
                &input.object_path(),
            )
            .await
            .map_err(ShardError::ObjectError)?;

            Transfer::transfer(
                self.ephemeral_store.object_store().clone(),
                self.persistent_store.object_store().clone(),
                &module_output.object_path(),
            )
            .await
            .map_err(ShardError::ObjectError)?;

            let download_metadata = self
                .persistent_store
                .download_metadata(module_output.object_path().clone())
                .await
                .map_err(ShardError::ObjectError)?;

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
