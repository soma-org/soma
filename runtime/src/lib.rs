use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use blobs::BlobPath;
use blobs::downloader::HttpBlobDownloader;
use burn::backend::{Cuda, NdArray, Wgpu};
use burn::tensor::TensorData;
use models::v1::ModelRunner;
use object_store::local::LocalFileSystem;
use tokio::sync::Semaphore;
use types::config::node_config::DeviceConfig;
use types::error::RuntimeResult;
use types::metadata::Manifest;
use types::parameters::HttpParameters;

pub mod v1;

pub use models::v1::modules::model::ModelConfig;

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
    /// Optional AES-256-CTR decryption keys, one per model. `None` means unencrypted.
    model_keys: Vec<Option<[u8; 32]>>,
    target: TensorData,
    seed: u64,
}

impl ManifestCompetitionInput {
    /// Create a new CompetitionInput.
    pub fn new(data: Manifest, models: Vec<Manifest>, target: TensorData, seed: u64) -> Self {
        let num_models = models.len();
        Self { data, models, model_keys: vec![None; num_models], target, seed }
    }

    /// Set decryption keys for encrypted model weights.
    pub fn with_model_keys(mut self, keys: Vec<Option<[u8; 32]>>) -> Self {
        self.model_keys = keys;
        self
    }

    /// Get the data manifest.
    pub fn data(&self) -> &Manifest {
        &self.data
    }

    /// Get the model entries (id + manifest).
    pub fn models(&self) -> &[Manifest] {
        &self.models
    }

    /// Get the decryption keys.
    pub fn model_keys(&self) -> &[Option<[u8; 32]>] {
        &self.model_keys
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

/// Create a `RuntimeV1` with the specified burn backend, returned as `Arc<dyn RuntimeAPI>`.
///
/// All backends are always compiled in. CUDA requires the NVIDIA CUDA toolkit
/// to be installed at runtime; it will error when initializing if unavailable.
pub fn build_runtime(
    device: &DeviceConfig,
    data_dir: &Path,
    model_config: ModelConfig,
) -> anyhow::Result<Arc<dyn RuntimeAPI>> {
    let store = Arc::new(
        LocalFileSystem::new_with_prefix(data_dir)
            .map_err(|e| anyhow::anyhow!("Failed to create local file system store: {e}"))?,
    );

    let http_params = HttpParameters::default();
    let semaphore = Arc::new(Semaphore::new(10));
    let chunk_size = 5 * 1024 * 1024; // 5MB
    let ns_per_byte = http_params.nanoseconds_per_byte as u16;

    let downloader = Arc::new(
        HttpBlobDownloader::new(&http_params, store.clone(), semaphore, chunk_size, ns_per_byte)
            .map_err(|e| anyhow::anyhow!("Failed to create blob downloader: {e}"))?,
    );

    match device {
        DeviceConfig::Cpu => {
            let model = Arc::new(ModelRunner::<NdArray>::new(model_config, Default::default(), 4));
            Ok(Arc::new(v1::RuntimeV1::new(store, downloader, 0, Default::default(), model)))
        }
        DeviceConfig::Wgpu => {
            let model = Arc::new(ModelRunner::<Wgpu>::new(model_config, Default::default(), 4));
            Ok(Arc::new(v1::RuntimeV1::new(store, downloader, 0, Default::default(), model)))
        }
        DeviceConfig::Cuda => {
            let model = Arc::new(ModelRunner::<Cuda>::new(model_config, Default::default(), 4));
            Ok(Arc::new(v1::RuntimeV1::new(store, downloader, 0, Default::default(), model)))
        }
    }
}
