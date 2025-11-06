use crate::inference::core_processor::InferenceCoreProcessor;
use crate::inference::messaging::InferenceService;
use crate::inference::module::ModuleClient;
use crate::inference::InferenceInput;
use async_trait::async_trait;
use bytes::Bytes;
use objects::{EphemeralStore, PersistentStore};
use tokio_util::sync::CancellationToken;
use types::actors::ActorHandle;
use types::error::{InferenceError, InferenceResult};

pub struct InferenceNetworkService<P: PersistentStore, E: EphemeralStore, M: ModuleClient> {
    core_processor: ActorHandle<InferenceCoreProcessor<P, E, M>>,
}

impl<P: PersistentStore, E: EphemeralStore, M: ModuleClient> InferenceNetworkService<P, E, M> {
    pub fn new(core_processor: ActorHandle<InferenceCoreProcessor<P, E, M>>) -> Self {
        Self { core_processor }
    }
}

#[async_trait]
impl<P: PersistentStore, E: EphemeralStore, M: ModuleClient> InferenceService
    for InferenceNetworkService<P, E, M>
{
    async fn handle_inference(&self, input_bytes: Bytes) -> InferenceResult<Bytes> {
        let inference_input: InferenceInput =
            bcs::from_bytes(&input_bytes).map_err(InferenceError::MalformedType)?;

        let inference_output = self
            .core_processor
            .process(inference_input, CancellationToken::new())
            .await
            .map_err(|e| InferenceError::CoreProcessorError(e.to_string()))?;

        Ok(Bytes::copy_from_slice(
            &bcs::to_bytes(&inference_output).map_err(InferenceError::SerializationFailure)?,
        ))
    }
}
