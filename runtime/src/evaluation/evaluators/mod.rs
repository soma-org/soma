use async_trait::async_trait;
use std::time::Duration;
use types::{
    error::EvaluationResult,
};
pub mod v1;
pub mod mock;

#[async_trait]
pub trait EvaluatorAPI: Send + Sync + Sized + 'static {
    async fn call(
        &self,
        input: EvaluationInput,
        timeout: Duration,
    ) -> EvaluationResult<EvaluationOutput>;
}
