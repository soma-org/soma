use std::sync::Arc;

use crate::{CompetitionAPI, CompetitionInput, CompetitionOutput};
use async_trait::async_trait;
use blobs::{BlobPath, downloader::BlobDownloader, loader::BlobLoader};
use burn::{
    Tensor, data::dataloader::DataLoaderBuilder, prelude::Backend, tensor::linalg::l2_norm,
};
use models::{
    ModelAPI, ModelOutput,
    v1::{
        data::{batcher::ByteSequenceBatcher, dataset::ByteSequenceDataset},
        modules::model::ModelConfig,
    },
};
use types::{
    committee::EpochId,
    error::{RuntimeError, RuntimeResult},
    metadata::{ManifestAPI, MetadataAPI},
    model::ModelId,
};

pub(crate) struct CompetitonV1<B: Backend, S: BlobLoader, D: BlobDownloader> {
    store: Arc<S>,
    downloader: D,
    epoch: EpochId,
    device: B::Device,
    max_seq_len: usize,
    batch_size: usize,
    num_workers: usize,
    model_config: ModelConfig,
}

#[async_trait]
impl<B: Backend, S: BlobLoader, D: BlobDownloader> CompetitionAPI for CompetitonV1<B, S, D> {
    async fn run(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput> {
        let data_blob_path = BlobPath::Data(self.epoch, input.data.metadata().checksum());

        self.downloader
            .download(&input.data, data_blob_path.clone())
            .await
            .map_err(RuntimeError::BlobError)?;

        let data_buffer =
            self.store.get_buffer(data_blob_path).await.map_err(RuntimeError::BlobError)?;

        let buffer = Arc::from(data_buffer.as_ref());

        let dataset = ByteSequenceDataset::new(self.max_seq_len, buffer);
        let batcher = ByteSequenceBatcher::new();
        let data_loader = DataLoaderBuilder::new(batcher)
            .batch_size(self.batch_size)
            .num_workers(self.num_workers)
            .build(dataset);

        let model = self.model_config.init(&self.device);

        if input.models.is_empty() {
            return Err(RuntimeError::ModelError("no models provided".into()));
        }

        let target: Tensor<B, 1> = Tensor::from_data(input.target, &self.device);
        let target_unit = l2_norm(target, 0);

        let mut best: Option<(ModelId, Tensor<B, 1>, f32)> = None;

        for (model_id, manifest) in input.models.iter() {
            let model_blob_path = BlobPath::Weights(self.epoch, manifest.metadata().checksum());
            self.downloader
                .download(manifest, model_blob_path.clone())
                .await
                .map_err(RuntimeError::BlobError)?;

            let output: ModelOutput<B> = model
                .call(data_loader.clone(), &model_blob_path)
                .map_err(|e| RuntimeError::ModelError(e.to_string()))?;

            let loss_val = output.loss.into_data().to_vec::<f32>().unwrap()[0];

            let is_better = best.as_ref().map_or(true, |(_, _, best)| loss_val < *best);

            if is_better {
                best = Some((*model_id, output.embedding, loss_val));
            }
        }

        let (winner, embedding, loss_val) =
            best.expect("models is non-empty, so best is always set");

        let emb_unit = l2_norm(embedding, 0);
        let cos_dist = emb_unit.clone().dot(target_unit).neg().add_scalar(1.0);
        let loss_score = Tensor::<B, 1>::from_floats([loss_val], &self.device);

        Ok(CompetitionOutput {
            winner,
            loss_score: loss_score.into_data_async().await,
            embedding: emb_unit.into_data_async().await,
            distance: cos_dist.into_data_async().await,
        })
    }
}
