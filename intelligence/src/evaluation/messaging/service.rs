use super::EvaluationService;
use crate::evaluation::core::{pipeline::CoreProcessor, safetensor_buffer::SafetensorBuffer};
use async_trait::async_trait;
use bytes::Bytes;
use objects::storage::ObjectStorage;
use tokio_util::sync::CancellationToken;
use types::error::{EvaluationError, EvaluationResult};
use types::{
    actors::ActorHandle,
    evaluation::{EmbeddingDigest, EvaluationInput, EvaluationOutput, EvaluationOutputV1, ScoreV1},
    shard_crypto::digest::Digest,
};

pub struct Evaluator<S: ObjectStorage + SafetensorBuffer> {
    core_processor: ActorHandle<CoreProcessor<S>>,
}

impl<S: ObjectStorage + SafetensorBuffer> Evaluator<S> {
    pub fn new(core_processor: ActorHandle<CoreProcessor<S>>) -> Self {
        Self { core_processor }
    }
}

#[async_trait]
impl<S: ObjectStorage + SafetensorBuffer> EvaluationService for Evaluator<S> {
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

pub struct MockEvaluationService {}

impl MockEvaluationService {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl EvaluationService for MockEvaluationService {
    async fn handle_evaluation(&self, input_bytes: Bytes) -> EvaluationResult<Bytes> {
        let _input: EvaluationInput =
            bcs::from_bytes(&input_bytes).map_err(EvaluationError::MalformedType)?;

        // TODO: do something with the input

        let score = ScoreV1::new(rand::random());
        let summary_embedding: EmbeddingDigest = Digest::new(&vec![1, 1, 1]).unwrap();

        let output = EvaluationOutput::V1(EvaluationOutputV1::new(score, summary_embedding));

        let output_bytes =
            Bytes::from(bcs::to_bytes(&output).map_err(EvaluationError::MalformedType)?);
        Ok(output_bytes)
    }
}
