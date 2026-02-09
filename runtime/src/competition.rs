// input manifest, models (modelId, manifests) -> spits out an embedding, score, and winning model index

use std::sync::Arc;

use crate::{CompetitionAPI, CompetitionInput, CompetitionOutput};
use async_trait::async_trait;
use blobs::{BlobPath, downloader::BlobDownloader, readers::url::BlobClient};
use burn::{
    Tensor,
    data::dataloader::DataLoaderBuilder,
    prelude::{Backend, ToElement},
    store::{ModuleSnapshot, SafetensorsStore},
    tensor::linalg::l2_norm,
};
use memmap2::MmapOptions;
use models::{
    ModelAPI, ModelOutput,
    v1::{
        data::{batcher::ByteSequenceBatcher, dataset::ByteSequenceDataset},
        probe::ProbeConfig,
    },
};
use object_store::local::LocalFileSystem;
use tokio::fs::File;
use types::{
    committee::EpochId,
    error::{RuntimeError, RuntimeResult},
    metadata::{ManifestAPI, MetadataAPI},
    model::ModelId,
};

pub(crate) struct MockCompetition {}

pub(crate) struct CompetitonV1<B: Backend> {
    store: Arc<LocalFileSystem>,
    blob_client: BlobClient,
    downloader: BlobDownloader,
    epoch: EpochId,
    device: B::Device,
    max_seq_len: usize,
    batch_size: usize,
    num_workers: usize,
}

#[async_trait]
impl<B: Backend> CompetitionAPI for CompetitonV1<B> {
    async fn run(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput> {
        // download data if not already in store
        let url_reader = Arc::new(
            self.blob_client.get_reader(&input.data).await.map_err(RuntimeError::BlobError)?,
        );
        let data_blob_path = BlobPath::Data(self.epoch, input.data.metadata().checksum());

        self.downloader
            .download(
                url_reader,
                self.store.clone(),
                data_blob_path.clone(),
                input.data.metadata().clone(),
            )
            .await
            .map_err(RuntimeError::BlobError)?;

        let fs_path = self
            .store
            .path_to_filesystem(&data_blob_path.path())
            .map_err(|e| RuntimeError::StorageFailure(e.to_string()))?;

        let file =
            File::open(fs_path).await.map_err(|e| RuntimeError::StorageFailure(e.to_string()))?;

        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| RuntimeError::StorageFailure(e.to_string()))?;

        let buffer = Arc::from(mmap.as_ref());

        let dataset = ByteSequenceDataset::new(self.max_seq_len, buffer);
        let batcher = ByteSequenceBatcher::new();
        let data_loader = DataLoaderBuilder::new(batcher)
            .batch_size(self.batch_size)
            .num_workers(self.num_workers)
            .build(dataset);

        let mut model = ProbeConfig::new().init(&self.device);

        let mut embeddings: Vec<Tensor<B, 1>> = Vec::new();
        let mut losses: Vec<Tensor<B, 1>> = Vec::new();
        let mut model_ids: Vec<ModelId> = Vec::new();

        for (model_id, manifest) in input.models.iter() {
            let url_reader = Arc::new(
                self.blob_client.get_reader(manifest).await.map_err(RuntimeError::BlobError)?,
            );
            let model_blob_path = BlobPath::Weights(self.epoch, manifest.metadata().checksum());
            self.downloader
                .download(
                    url_reader,
                    self.store.clone(),
                    model_blob_path.clone(),
                    manifest.metadata().clone(),
                )
                .await
                .map_err(RuntimeError::BlobError)?;

            let mut store = SafetensorsStore::from_file(
                self.store
                    .path_to_filesystem(&model_blob_path.path())
                    .map_err(RuntimeError::ObjectStoreError)?,
            );
            model.load_from(&mut store).unwrap();
            let model_output: ModelOutput<B> = model.call(data_loader.clone()).unwrap();

            embeddings.push(model_output.embedding);
            losses.push(model_output.loss);
            model_ids.push(*model_id);
        }

        Ok(finalize(model_ids, losses, embeddings, input.target.into()).await)
    }
}

async fn finalize<B: Backend>(
    models: Vec<ModelId>,
    losses: Vec<Tensor<B, 1>>,
    embeddings: Vec<Tensor<B, 1>>,
    target: Tensor<B, 1>,
) -> CompetitionOutput {
    const EPS: f32 = 1e-8;
    let losses_st = Tensor::stack(losses, 0);
    let embs_st = Tensor::stack(embeddings, 0);
    let weights = (losses_st.clone() + EPS).recip();
    let total_w = weights.clone().sum_dim(0);
    let final_emb = (embs_st * weights).sum_dim(0) / total_w.squeeze();

    let final_unit = l2_norm(final_emb, 0);
    let target_unit = l2_norm(target, 0);

    let cos_sim = final_unit.clone().dot(target_unit);
    let cos_dist: Tensor<B, 1> = 1.0 - cos_sim;

    let (_min_loss, winner_idx) = losses_st.min_dim_with_indices(0); // or min_dim_with_indices

    let winner_idx = winner_idx.into_scalar_async().await.to_i64();
    let winner_idx: usize = winner_idx.try_into().unwrap_or(0);
    let winner_id = models[winner_idx];

    CompetitionOutput {
        winner: winner_id,
        embedding: final_unit.into_data_async().await,
        distance: cos_dist.into_data_async().await,
    }
}
