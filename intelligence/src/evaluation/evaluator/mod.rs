use async_trait::async_trait;
use std::time::Duration;
use types::{
    error::EvaluationResult,
    evaluation::{EvaluationInput, EvaluationOutput},
};
pub mod evaluator;
pub mod mock;

#[async_trait]
pub trait EvaluatorClient: Send + Sync + Sized + 'static {
    async fn call(
        &self,
        input: EvaluationInput,
        timeout: Duration,
    ) -> EvaluationResult<EvaluationOutput>;
}
