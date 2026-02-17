use std::sync::Arc;

use arrgen::normal_array;
use burn::{
    data::dataloader::DataLoader,
    nn::loss::CrossEntropyLossConfig,
    prelude::Backend,
    store::{ModuleSnapshot, SafetensorsStore},
    tensor::Tensor,
};
use types::error::ModelError;

use crate::{
    ModelAPI, ModelOutput,
    running_mean::RunningMean,
    tensor_conversions::IntoTensorData,
    v1::{
        data::{batcher::ByteSequenceBatch, build_data_loader, dataset::PAD_TOKEN_ID},
        modules::{model::ModelConfig, sig_reg::SIGRegConfig},
    },
};

pub mod data;
pub mod modules;

const V1_EMBEDDING_DIM: usize = 2048;
const V1_NUM_HEADS: usize = 8;
const V1_NUM_LAYERS: usize = 32;
const V1_MAX_SEQ_LEN: usize = 8192;
const V1_PWFF_HIDDEN_DIM: usize = V1_EMBEDDING_DIM * 4;
const V1_MAX_WAVELENGTH: f32 = 10_000.0;
const V1_SCALE_FACTOR: f32 = 1.0;
const V1_VOCAB_SIZE: usize = 256 + 8;
const V1_SIG_REG_T_MAX: f64 = 3.0;
const V1_SIG_REG_SLICES: usize = 1024;
const V1_SIG_REG_POINTS: usize = 17;
const V1_SIG_REG_COEFFICIENT: f64 = 0.02;
const V1_BATCH_SIZE: usize = 32;

pub struct ModelRunner<B: Backend> {
    config: ModelConfig,
    device: B::Device,
    max_seq_len: usize,
    batch_size: usize,
    num_data_workers: usize,
    sig_reg_config: SIGRegConfig,
    sig_reg_coefficient: f64,
}

impl<B: Backend> ModelRunner<B> {
    pub fn new(config: ModelConfig, device: B::Device, num_data_workers: usize) -> Self {
        Self {
            config,
            device,
            max_seq_len: V1_MAX_SEQ_LEN,
            batch_size: V1_BATCH_SIZE,
            num_data_workers,
            sig_reg_config: SIGRegConfig::new(),
            sig_reg_coefficient: V1_SIG_REG_COEFFICIENT,
        }
    }

    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

#[async_trait::async_trait]
impl<B: Backend> ModelAPI for ModelRunner<B> {
    type Data = Arc<dyn DataLoader<B, ByteSequenceBatch<B>> + Send + Sync + 'static>;
    type Backend = B;

    async fn dataloader(&self, buffer: Arc<[u8]>) -> Self::Data {
        build_data_loader::<B>(buffer, self.max_seq_len, self.batch_size, self.num_data_workers)
    }

    async fn eval(
        &self,
        data: Self::Data,
        weights: SafetensorsStore,
        seed: u64,
    ) -> types::error::ModelResult<ModelOutput<Self::Backend>> {
        let mut weights = weights;
        let mut model = self.config.init::<B>(&self.device);
        let apply_results =
            model.load_from(&mut weights).map_err(ModelError::SafeTensorStoreError)?;

        if !apply_results.is_success() {
            return Err(ModelError::ApplyError);
        }

        let cross_entropy = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD_TOKEN_ID as usize]))
            .init(&self.device);
        let sig_reg = self.sig_reg_config.init::<B>(&self.device);

        let mut embeddings = RunningMean::new();
        let mut losses = RunningMean::new();

        for (idx, batch) in data.iter().enumerate() {
            let representations = model.encode(batch.token_ids.clone(), batch.pos_ids.clone());
            let logits = model.predict(representations.clone());

            let noise_data = normal_array(
                seed + idx as u64,
                &[self.config.embedding_dim, sig_reg.slices],
                0.0,
                1.0,
            )
            .to_tensor_data()
            .map_err(|e| ModelError::EmptyData(e.to_string()))?;
            let noise: Tensor<B, 2> = Tensor::from_data(noise_data, &self.device);
            let sig_reg_loss = sig_reg.forward(representations.clone(), noise);

            // mean embedding: [batch, seq, embed] -> [embed]
            let embedding =
                representations.mean_dim(1).squeeze_dim::<2>(1).mean_dim(0).squeeze_dim::<1>(0);

            // cross entropy: flatten [batch, seq, vocab] -> [batch*seq, vocab]
            let [batch_size, seq, vocab] = logits.dims();
            let logits_flat = logits.reshape([batch_size * seq, vocab]);
            let targets_flat = batch.targets.reshape([batch_size * seq]);
            let ce_loss = cross_entropy.forward(logits_flat, targets_flat).mean();
            let loss = ce_loss + (sig_reg_loss * self.sig_reg_coefficient);

            embeddings.add(embedding);
            losses.add(loss);
        }

        let embedding = embeddings
            .value()
            .ok_or_else(|| ModelError::EmptyData("data loader produced no batches".into()))?;
        let loss = losses
            .value()
            .ok_or_else(|| ModelError::EmptyData("data loader produced no batches".into()))?;
        Ok(ModelOutput { embedding, loss })
    }
}
