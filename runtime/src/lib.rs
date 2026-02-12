use async_trait::async_trait;
use blobs::BlobPath;
use burn::{Tensor, data::dataloader::DataLoader, prelude::Backend, tensor::TensorData};
use std::sync::Arc;
use types::{error::RuntimeResult, metadata::Manifest, model::ModelId};
pub mod competition;

/// Input for running a competition evaluation.
pub struct CompetitionInput {
    data: Manifest,
    models: Vec<(ModelId, Manifest)>,
    target: TensorData,
}

impl CompetitionInput {
    /// Create a new CompetitionInput.
    pub fn new(data: Manifest, models: Vec<(ModelId, Manifest)>, target: TensorData) -> Self {
        Self { data, models, target }
    }

    /// Get the data manifest.
    pub fn data(&self) -> &Manifest {
        &self.data
    }

    /// Get the model manifests.
    pub fn models(&self) -> &[(ModelId, Manifest)] {
        &self.models
    }

    /// Get the target embedding.
    pub fn target(&self) -> &TensorData {
        &self.target
    }
}

/// Output from running a competition evaluation.
pub struct CompetitionOutput {
    winner: ModelId,
    loss_score: TensorData,
    embedding: TensorData,
    distance: TensorData,
}

impl CompetitionOutput {
    /// Create a new CompetitionOutput.
    pub fn new(
        winner: ModelId,
        loss_score: TensorData,
        embedding: TensorData,
        distance: TensorData,
    ) -> Self {
        Self { winner, loss_score, embedding, distance }
    }

    /// Get the winning model ID.
    pub fn winner(&self) -> ModelId {
        self.winner
    }

    /// Get the loss score.
    pub fn loss_score(&self) -> &TensorData {
        &self.loss_score
    }

    /// Get the computed embedding.
    pub fn embedding(&self) -> &TensorData {
        &self.embedding
    }

    /// Get the computed distance (scalar TensorData).
    pub fn distance(&self) -> &TensorData {
        &self.distance
    }
}

#[async_trait]
pub trait CompetitionAPI: Send + Sync + 'static {
    async fn run(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput>;
}
