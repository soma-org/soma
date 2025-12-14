use async_trait::async_trait;
use object_store::ObjectStore;
use std::sync::Arc;
use types::{
    error::EvaluationResult,
    evaluation::{EvaluationInput, EvaluationOutput},
};
pub mod core;
pub mod core_processor;

#[async_trait]
pub trait EvaluatorClient: Send + Sync + Sized + 'static {
    async fn call(
        &self,
        input: EvaluationInput,
        storage: Arc<dyn ObjectStore>,
    ) -> EvaluationResult<EvaluationOutput>;
}

pub struct MockEvaluator {
    output: EvaluationOutput,
}

impl MockEvaluator {
    pub fn new(output: EvaluationOutput) -> Self {
        Self { output }
    }
}

#[async_trait]
impl EvaluatorClient for MockEvaluator {
    async fn call(
        &self,
        input: EvaluationInput,
        storage: Arc<dyn ObjectStore>,
    ) -> EvaluationResult<EvaluationOutput> {
        Ok(self.output.clone())
    }
}
