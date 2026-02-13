use async_trait::async_trait;
use burn::tensor::TensorData;
use types::{error::RuntimeResult, metadata::Manifest};
pub mod v1;
use blobs::BlobPath;

/// Input for running a competition evaluation.
pub struct CompetitionInput {
    data: BlobPath,
    models: Vec<BlobPath>,
    target: TensorData,
    seed: u64,
}

impl CompetitionInput {
    /// Create a new CompetitionInput.
    pub fn new(data: BlobPath, models: Vec<BlobPath>, target: TensorData, seed: u64) -> Self {
        Self { data, models, target, seed }
    }

    /// Get the data manifest.
    pub fn data(&self) -> &BlobPath {
        &self.data
    }

    /// Get the model entries (id + manifest).
    pub fn models(&self) -> &[BlobPath] {
        &self.models
    }

    /// Get the target embedding.
    pub fn target(&self) -> &TensorData {
        &self.target
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

pub struct ManifestCompetitionInput {
    data: Manifest,
    models: Vec<Manifest>,
    target: TensorData,
    seed: u64,
}

impl ManifestCompetitionInput {
    /// Create a new CompetitionInput.
    pub fn new(data: Manifest, models: Vec<Manifest>, target: TensorData, seed: u64) -> Self {
        Self { data, models, target, seed }
    }

    /// Get the data manifest.
    pub fn data(&self) -> &Manifest {
        &self.data
    }

    /// Get the model entries (id + manifest).
    pub fn models(&self) -> &[Manifest] {
        &self.models
    }

    /// Get the target embedding.
    pub fn target(&self) -> &TensorData {
        &self.target
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// Output from running a competition evaluation.
#[derive(Debug)]
pub struct CompetitionOutput {
    winner: usize,
    loss_score: TensorData,
    embedding: TensorData,
    distance: TensorData,
}

impl CompetitionOutput {
    /// Create a new CompetitionOutput.
    pub fn new(
        winner: usize,
        loss_score: TensorData,
        embedding: TensorData,
        distance: TensorData,
    ) -> Self {
        Self { winner, loss_score, embedding, distance }
    }

    /// Get the winning model index.
    pub fn winner(&self) -> usize {
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
pub trait RuntimeAPI: Send + Sync + 'static {
    async fn competition(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput>;
    async fn download_manifest(&self, manifest: &Manifest, path: &BlobPath) -> RuntimeResult<()>;
    async fn manifest_competition(
        &self,
        input: ManifestCompetitionInput,
    ) -> RuntimeResult<CompetitionOutput>;
}
