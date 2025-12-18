use super::EvaluationService;
use crate::evaluation::core_processor::EvaluationCoreProcessor;
use crate::evaluation::evaluator::EvaluatorClient;
use async_trait::async_trait;
use bytes::Bytes;
use object_store::ObjectStore;
use objects::stores::{EphemeralStore, PersistentStore};
use tokio_util::sync::CancellationToken;
use types::error::{EvaluationError, EvaluationResult};
use types::{actors::ActorHandle, evaluation::EvaluationInput};

pub struct EvaluationNetworkService<
    PS: ObjectStore,
    ES: ObjectStore,
    P: PersistentStore<PS>,
    E: EphemeralStore<ES>,
    C: EvaluatorClient,
> {
    core_processor: ActorHandle<EvaluationCoreProcessor<PS, ES, P, E, C>>,
}

impl<
        PS: ObjectStore,
        ES: ObjectStore,
        P: PersistentStore<PS>,
        E: EphemeralStore<ES>,
        C: EvaluatorClient,
    > EvaluationNetworkService<PS, ES, P, E, C>
{
    pub fn new(core_processor: ActorHandle<EvaluationCoreProcessor<PS, ES, P, E, C>>) -> Self {
        Self { core_processor }
    }
}

#[async_trait]
impl<
        PS: ObjectStore,
        ES: ObjectStore,
        P: PersistentStore<PS>,
        E: EphemeralStore<ES>,
        C: EvaluatorClient,
    > EvaluationService for EvaluationNetworkService<PS, ES, P, E, C>
{
    async fn handle_evaluation(&self, input_bytes: Bytes) -> EvaluationResult<Bytes> {
        let evaluation_input: EvaluationInput =
            bcs::from_bytes(&input_bytes).map_err(EvaluationError::MalformedType)?;

        let evaluation_output = self
            .core_processor
            .process(evaluation_input, CancellationToken::new())
            .await
            .unwrap(); // TODO: handle this error better

        let serialized_evaluation_output = Bytes::copy_from_slice(
            &bcs::to_bytes(&evaluation_output).map_err(EvaluationError::SerializationFailure)?,
        );
        Ok(serialized_evaluation_output)
    }
}
