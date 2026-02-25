// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{CompetitionInput, CompetitionOutput, ManifestCompetitionInput, RuntimeAPI};
use aes::Aes256;
use async_trait::async_trait;
use blobs::{BlobPath, downloader::BlobDownloader, loader::BlobLoader};
use burn::{Tensor, prelude::Backend};
use ctr::Ctr128BE;
use ctr::cipher::{KeyIvInit, StreamCipher};
use models::{
    ModelAPI, ModelOutput, cosine_distance::cosine_distance, select_best::select_best_model,
};
use object_store::{ObjectStore, PutPayload};
use tokio::task::JoinSet;
use types::{
    committee::EpochId,
    error::{BlobError, RuntimeError, RuntimeResult},
    metadata::{Manifest, ManifestAPI, MetadataAPI},
};

/// Decrypt data in-place with AES-256-CTR (zero IV).
fn decrypt_aes256_ctr(data: &mut [u8], key: &[u8; 32]) {
    let iv = [0u8; 16];
    let mut cipher = Ctr128BE::<Aes256>::new(key.into(), &iv.into());
    cipher.apply_keystream(data);
}

pub struct RuntimeV1<B, S, D, M>
where
    B: Backend,
    S: ObjectStore + BlobLoader,
    D: BlobDownloader<Store = S>,
    M: ModelAPI<Backend = B>,
{
    store: Arc<S>,
    downloader: Arc<D>,
    epoch: EpochId,
    device: B::Device,
    model: Arc<M>,
}

impl<B, S, D, M> RuntimeV1<B, S, D, M>
where
    B: Backend,
    S: ObjectStore + BlobLoader,
    D: BlobDownloader<Store = S>,
    M: ModelAPI<Backend = B>,
{
    pub fn new(
        store: Arc<S>,
        downloader: Arc<D>,
        epoch: EpochId,
        device: B::Device,
        model: Arc<M>,
    ) -> Self {
        Self { store, downloader, epoch, device, model }
    }
}

#[async_trait]
impl<B, S, D, M> RuntimeAPI for RuntimeV1<B, S, D, M>
where
    B: Backend,
    S: ObjectStore + BlobLoader,
    D: BlobDownloader<Store = S>,
    M: ModelAPI<Backend = B>,
{
    async fn competition(&self, input: CompetitionInput) -> RuntimeResult<CompetitionOutput> {
        let data_buffer =
            self.store.get_buffer(input.data()).await.map_err(RuntimeError::BlobError)?;
        let buffer = Arc::from(data_buffer.as_ref());

        let dataloader = self.model.dataloader(buffer).await;

        let mut outputs = Vec::with_capacity(input.models.len());
        for model_path in &input.models {
            let safetensor_store =
                self.store.safetensor_store(model_path).await.map_err(RuntimeError::BlobError)?;
            let output: ModelOutput<B> = self
                .model
                .eval(dataloader.clone(), safetensor_store, input.seed())
                .await
                .map_err(|e| RuntimeError::ModelError(e.to_string()))?;
            outputs.push(output);
        }

        let best_idx = select_best_model(&outputs)
            .ok_or(RuntimeError::ValidationError("select best model returned None".to_string()))?;
        let best = &outputs[best_idx];

        let target: Tensor<B, 1> = Tensor::from_data(input.target, &self.device);
        let distance = cosine_distance(best.embedding.clone(), target);
        let loss_score = best.loss.clone();

        Ok(CompetitionOutput {
            winner: best_idx,
            loss_score: loss_score.into_data_async().await,
            embedding: best.embedding.clone().into_data_async().await,
            distance: distance.into_data_async().await,
        })
    }

    async fn download_manifest(&self, manifest: &Manifest, path: &BlobPath) -> RuntimeResult<()> {
        self.downloader.download(manifest, path.clone()).await.map_err(RuntimeError::BlobError)?;
        Ok(())
    }

    async fn manifest_competition(
        &self,
        input: ManifestCompetitionInput,
    ) -> RuntimeResult<CompetitionOutput> {
        let data_path = BlobPath::Data(self.epoch, input.data().metadata().checksum());

        let mut model_paths = Vec::with_capacity(input.models().len());
        for manifest in input.models() {
            model_paths.push(BlobPath::Weights(self.epoch, manifest.metadata().checksum()));
        }

        let mut join_set = JoinSet::new();

        let downloader = self.downloader.clone();
        let data_manifest = input.data().clone();
        let data_path_clone = data_path.clone();
        join_set.spawn(async move { downloader.download(&data_manifest, data_path_clone).await });

        for (i, manifest) in input.models().iter().enumerate() {
            let downloader = self.downloader.clone();
            let manifest = manifest.clone();
            let path = model_paths[i].clone();
            join_set.spawn(async move { downloader.download(&manifest, path).await });
        }

        while let Some(result) = join_set.join_next().await {
            result
                .map_err(|e| RuntimeError::ModelError(e.to_string()))?
                .map_err(RuntimeError::BlobError)?;
        }

        // Decrypt model weights if decryption keys are provided.
        for (i, key) in input.model_keys().iter().enumerate() {
            if let Some(key) = key {
                let path = &model_paths[i];
                let get_result = self
                    .store
                    .get(&path.path())
                    .await
                    .map_err(|e| RuntimeError::BlobError(BlobError::ObjectStoreError(e)))?;
                let encrypted = get_result
                    .bytes()
                    .await
                    .map_err(|e| RuntimeError::BlobError(BlobError::ObjectStoreError(e)))?;
                let mut decrypted = encrypted.to_vec();
                decrypt_aes256_ctr(&mut decrypted, key);
                self.store
                    .put(&path.path(), PutPayload::from(decrypted))
                    .await
                    .map_err(|e| RuntimeError::BlobError(BlobError::ObjectStoreError(e)))?;
            }
        }

        let competition_input =
            CompetitionInput::new(data_path, model_paths, input.target().clone(), input.seed());
        self.competition(competition_input).await
    }
}
