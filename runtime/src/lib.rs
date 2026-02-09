use async_trait::async_trait;
use blobs::BlobPath;
use burn::{Tensor, data::dataloader::DataLoader, prelude::Backend, tensor::TensorData};
use std::sync::Arc;
use types::{error::RuntimeResult, metadata::Manifest, model::ModelId};
pub mod competition;

pub struct CompetitionInput {
    data: Manifest,
    models: Vec<(ModelId, Manifest)>,
    target: TensorData,
}

pub struct CompetitionOutput {
    winner: ModelId,
    embedding: TensorData,
    distance: TensorData,
}

#[async_trait]
pub trait CompetitionAPI: Send + Sync + 'static {
    async fn run(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput>;
}
