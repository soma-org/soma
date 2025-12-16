use async_trait::async_trait;
use object_store::ObjectStore;
use std::sync::Arc;
use types::{
    error::EvaluationResult,
    evaluation::{
        EvaluationInput, EvaluationInputAPI, EvaluationOutput, EvaluationOutputAPI,
        EvaluationOutputV1,
    },
};
pub mod core;
pub mod core_processor;
pub mod messaging;

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
        let EvaluationOutput::V1(v1) = self.output.clone();

        if input.target_embedding().is_none() {
            Ok(EvaluationOutput::V1(EvaluationOutputV1::new(
                v1.evaluation_scores().clone(),
                v1.summary_embedding().clone(),
                v1.sampled_embedding().clone(),
                None,
            )))
        } else {
            Ok(self.output.clone())
        }
    }
}
