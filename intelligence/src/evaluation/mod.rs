use async_trait::async_trait;
use enum_dispatch::enum_dispatch;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use types::{
    error::EvaluationResult,
    evaluation::{EmbeddingDigest, Score, ScoreV1},
    metadata::ObjectPath,
};
pub mod core;
pub mod core_processor;
pub mod messaging;

#[enum_dispatch]
pub(crate) trait EvaluatorInputAPI {
    fn input_object_path(&self) -> &ObjectPath;
    fn embedding_object_path(&self) -> &ObjectPath;
    fn probe_object_path(&self) -> &ObjectPath;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct EvaluatorInputV1 {
    input_object_path: ObjectPath,
    embedding_object_path: ObjectPath,
    probe_object_path: ObjectPath,
}

impl EvaluatorInputV1 {
    pub fn new(
        input_object_path: ObjectPath,
        embedding_object_path: ObjectPath,
        probe_object_path: ObjectPath,
    ) -> Self {
        Self {
            input_object_path,
            embedding_object_path,
            probe_object_path,
        }
    }
}

impl EvaluatorInputAPI for EvaluatorInputV1 {
    fn input_object_path(&self) -> &ObjectPath {
        &self.input_object_path
    }
    fn embedding_object_path(&self) -> &ObjectPath {
        &self.embedding_object_path
    }
    fn probe_object_path(&self) -> &ObjectPath {
        &self.probe_object_path
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(EvaluatorInputAPI)]
pub enum EvaluatorInput {
    V1(EvaluatorInputV1),
}

#[enum_dispatch]
pub trait EvaluatorOutputAPI {
    fn score(&self) -> &Score;
    fn summary_digest(&self) -> &EmbeddingDigest;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvaluatorOutputV1 {
    score: Score,
    summary_digest: EmbeddingDigest,
}

impl EvaluatorOutputV1 {
    pub fn new(score: Score, summary_digest: EmbeddingDigest) -> Self {
        Self {
            score,
            summary_digest,
        }
    }
}
impl EvaluatorOutputAPI for EvaluatorOutputV1 {
    fn score(&self) -> &Score {
        &self.score
    }
    fn summary_digest(&self) -> &EmbeddingDigest {
        &self.summary_digest
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(EvaluatorOutputAPI)]
pub enum EvaluatorOutput {
    V1(EvaluatorOutputV1),
}

#[async_trait]
pub trait EvaluatorClient: Send + Sync + Sized + 'static {
    async fn call(
        &self,
        input: EvaluatorInput,
        storage: Arc<dyn ObjectStore>,
    ) -> EvaluationResult<EvaluatorOutput>;
}

pub struct MockEvaluator {
    score: Score,
}

impl MockEvaluator {
    pub fn new(score: Score) -> Self {
        Self { score }
    }
}

#[async_trait]
impl EvaluatorClient for MockEvaluator {
    async fn call(
        &self,
        input: EvaluatorInput,
        storage: Arc<dyn ObjectStore>,
    ) -> EvaluationResult<EvaluatorOutput> {
        Ok(EvaluatorOutput::V1(EvaluatorOutputV1::new(
            self.score.clone(),
            EmbeddingDigest::new_from_bytes(b"fill"),
        )))
    }
}
