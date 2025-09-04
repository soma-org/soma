use types::evaluation::{
    EvaluationInput, EvaluationOutput, EvaluationOutputV1, EvaluationScoreV1, SummaryEmbeddingV1,
};

use super::EvaluationService;
use async_trait::async_trait;
use bytes::Bytes;
use types::error::{EvaluationError, EvaluationResult};

pub struct MockEvaluationService {}

impl MockEvaluationService {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl EvaluationService for MockEvaluationService {
    async fn handle_evaluation(&self, input_bytes: Bytes) -> EvaluationResult<Bytes> {
        let input: EvaluationInput =
            bcs::from_bytes(&input_bytes).map_err(EvaluationError::MalformedType)?;

        // TODO: do something with the input

        let score = EvaluationScoreV1::new(rand::random());
        let summary_embedding = SummaryEmbeddingV1::new(vec![1, 1, 1]);

        let output = EvaluationOutput::V1(EvaluationOutputV1::new(score, summary_embedding));

        let output_bytes =
            Bytes::from(bcs::to_bytes(&output).map_err(EvaluationError::MalformedType)?);
        Ok(output_bytes)
    }
}
