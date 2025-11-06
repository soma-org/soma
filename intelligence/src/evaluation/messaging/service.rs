use super::EvaluationService;
use crate::evaluation::core_processor::EvaluationCoreProcessor;
use crate::evaluation::EvaluatorClient;
use async_trait::async_trait;
use bytes::Bytes;
use objects::{EphemeralStore, PersistentStore};
use tokio_util::sync::CancellationToken;
use types::error::{EvaluationError, EvaluationResult};
use types::{actors::ActorHandle, evaluation::EvaluationInput};

pub struct EvaluationNetworkService<P: PersistentStore, E: EphemeralStore, C: EvaluatorClient> {
    core_processor: ActorHandle<EvaluationCoreProcessor<P, E, C>>,
}

impl<P: PersistentStore, E: EphemeralStore, C: EvaluatorClient> EvaluationNetworkService<P, E, C> {
    pub fn new(core_processor: ActorHandle<EvaluationCoreProcessor<P, E, C>>) -> Self {
        Self { core_processor }
    }
}

#[async_trait]
impl<P: PersistentStore, E: EphemeralStore, C: EvaluatorClient> EvaluationService
    for EvaluationNetworkService<P, E, C>
{
    async fn handle_evaluation(&self, input_bytes: Bytes) -> EvaluationResult<Bytes> {
        let evaluation_input: EvaluationInput =
            bcs::from_bytes(&input_bytes).map_err(EvaluationError::MalformedType)?;

        let evaluation_output = self
            .core_processor
            .process(evaluation_input, CancellationToken::new())
            .await
            .unwrap();

        let serialized_evaluation_output = Bytes::copy_from_slice(
            &bcs::to_bytes(&evaluation_output).map_err(EvaluationError::SerializationFailure)?,
        );
        Ok(serialized_evaluation_output)
    }
}
