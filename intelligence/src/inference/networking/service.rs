use crate::inference::engine::InferenceEngineAPI;
use crate::inference::work_queue::{InferenceWorkQueue};
use crate::inference::networking::InferenceService;
use crate::inference::InferenceInput;
use async_trait::async_trait;
use bytes::Bytes;
use object_store::ObjectStore;
use objects::stores::{EphemeralStore, PersistentStore};
use tokio_util::sync::CancellationToken;
use types::actors::ActorHandle;
use types::error::{InferenceError, InferenceResult};

pub struct InferenceNetworkService<
    I: InferenceEngineAPI,
> {
    work_queue: ActorHandle<InferenceWorkQueue<I>>,
}

impl<
    I: InferenceEngineAPI,
    > InferenceNetworkService<I>
{
    pub fn new(work_queue: ActorHandle<InferenceWorkQueue<I>>) -> Self {
        Self { work_queue }
    }
}

#[async_trait]
impl<
    I: InferenceEngineAPI,

    > InferenceService for InferenceNetworkService<I>
{
    async fn handle_inference(&self, input_bytes: Bytes) -> InferenceResult<Bytes> {
        let inference_input: InferenceInput =
            bcs::from_bytes(&input_bytes).map_err(InferenceError::MalformedType)?;

        let inference_output = self
            .work_queue
            .process(inference_input, CancellationToken::new())
            .await
            .map_err(|e| InferenceError::CoreProcessorError(e.to_string()))?;

        Ok(Bytes::copy_from_slice(
            &bcs::to_bytes(&inference_output).map_err(InferenceError::SerializationFailure)?,
        ))
    }
}
