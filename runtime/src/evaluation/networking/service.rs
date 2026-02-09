use super::EvaluationService;
use crate::evaluation::evaluators::EvaluatorAPI;
use crate::evaluation::work_queue::{EvaluationWorkQueue};
use async_trait::async_trait;
use bytes::Bytes;
use tokio_util::sync::CancellationToken;
use types::error::{EvaluationError, EvaluationResult};
use types::{actors::ActorHandle, evaluation::EvaluationInput};

pub struct EvaluationNetworkService<
    C: EvaluatorAPI,
> {
    work_queue: ActorHandle<EvaluationWorkQueue<C>>,
}

impl<
        C: EvaluatorAPI,
    > EvaluationNetworkService< C>
{
    pub fn new(work_queue: ActorHandle<EvaluationWorkQueue<C>>) -> Self {
        Self { work_queue }
    }
}

#[async_trait]
impl<
        C: EvaluatorAPI,
    > EvaluationService for EvaluationNetworkService<C>
{
    async fn handle_evaluation(&self, input_bytes: Bytes) -> EvaluationResult<Bytes> {
        let evaluation_input: EvaluationInput =
            bcs::from_bytes(&input_bytes).map_err(EvaluationError::MalformedType)?;

        let evaluation_output = self
            .work_queue
            .process(evaluation_input, CancellationToken::new())
            .await
            .unwrap(); // TODO: handle this error better

        let serialized_evaluation_output = Bytes::copy_from_slice(
            &bcs::to_bytes(&evaluation_output).map_err(EvaluationError::SerializationFailure)?,
        );
        Ok(serialized_evaluation_output)
    }
}
