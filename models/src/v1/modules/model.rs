use std::sync::Arc;

use blobs::BlobPath;
use burn::{
    config::Config,
    data::dataloader::DataLoader,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
    },
    tensor::{Int, Tensor, backend::Backend},
};
use types::error::{ModelError, ModelResult};

use crate::{
    ModelAPI, ModelOutput,
    tensor_conversions::IntoTensorData,
    v1::{
        V1_EMBEDDING_DIM, V1_MAX_WAVELENGTH, V1_NUM_HEADS, V1_NUM_LAYERS, V1_PWFF_HIDDEN_DIM,
        V1_SCALE_FACTOR, V1_SIG_REG_POINTS, V1_SIG_REG_SLICES, V1_SIG_REG_T_MAX, V1_VOCAB_SIZE,
        data::{batcher::ByteSequenceBatch, dataset::PAD_TOKEN_ID},
        modules::{
            encoder::Encoder, encoder::EncoderConfig, sig_reg::SIGReg, sig_reg::SIGRegConfig,
        },
    },
};

#[derive(Config, Debug)]
pub struct ModelConfig {
    /// The size of the input and output features.
    #[config(default = "V1_EMBEDDING_DIM")]
    pub embedding_dim: usize,
    /// The size of the hidden inner pwff features.
    #[config(default = "V1_PWFF_HIDDEN_DIM")]
    pub pwff_hidden_dim: usize,
    /// The number of transformer layers.
    #[config(default = "V1_NUM_LAYERS")]
    pub num_layers: usize,
    /// The number of transformer heads.
    #[config(default = "V1_NUM_HEADS")]
    pub num_heads: usize,
    /// The probability that dropout occurs
    #[config(default = 0.0)]
    pub dropout_rate: f64,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub weight_initializer: Initializer,
    #[config(default = "V1_VOCAB_SIZE")]
    pub vocab_size: usize,
    /// The max wavelength for RoPE.
    #[config(default = "V1_MAX_WAVELENGTH")]
    pub max_wavelength: f32,
    /// The RoPE scale factor.
    #[config(default = "V1_SCALE_FACTOR")]
    pub scale_factor: f32,

    #[config(default = "V1_SIG_REG_T_MAX")]
    pub sig_reg_t_max: f64,
    #[config(default = "V1_SIG_REG_SLICES")]
    pub sig_reg_slices: usize,
    #[config(default = "V1_SIG_REG_POINTS")]
    pub sig_reg_points: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    embedding: Embedding<B>,
    encoder: Encoder<B>,
    final_norm: LayerNorm<B>,
    predictor: Linear<B>,
    sig_reg: SIGReg<B>,
}

impl ModelConfig {
    /// Initialize a new module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            embedding: EmbeddingConfig::new(self.vocab_size, self.embedding_dim)
                .with_initializer(self.weight_initializer.clone())
                .init(device),
            encoder: EncoderConfig::new()
                .with_embedding_dim(self.embedding_dim)
                .with_pwff_hidden_dim(self.pwff_hidden_dim)
                .with_num_layers(self.num_layers)
                .with_num_heads(self.num_heads)
                .with_dropout_rate(self.dropout_rate)
                .with_max_wavelength(self.max_wavelength)
                .with_scale_factor(self.scale_factor)
                .with_initializer(self.weight_initializer.clone())
                .init(device),

            final_norm: LayerNormConfig::new(self.embedding_dim).init(device),
            predictor: LinearConfig::new(self.embedding_dim, self.vocab_size)
                .with_initializer(self.weight_initializer.clone())
                .init(device),
            sig_reg: SIGRegConfig::new()
                .with_t_max(self.sig_reg_t_max)
                .with_slices(self.sig_reg_slices)
                .with_points(self.sig_reg_points)
                .init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, positions: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(tokens);
        let x = self.encoder.forward(x, positions);
        self.final_norm.forward(x)
    }

    pub fn predictor(&self, embeddings: Tensor<B, 3>) -> Tensor<B, 3> {
        self.predictor.forward(embeddings)
    }
}

impl<B: Backend> ModelAPI for Model<B> {
    type Data = Arc<dyn DataLoader<B, ByteSequenceBatch<B>> + Send + Sync + 'static>;
    type Backend = B;
    fn call(
        &self,
        data: Self::Data,
        weights: &BlobPath,
    ) -> ModelResult<ModelOutput<Self::Backend>> {
        let embedding_dim = self.embedding.weight.val().dims()[1];
        let loss_config =
            CrossEntropyLossConfig::new().with_pad_tokens(Some(vec![PAD_TOKEN_ID as usize]));

        let mut loss_fn = None;
        let mut mean_loss: Option<Tensor<B, 1>> = None;
        let mut mean_embedding: Option<Tensor<B, 1>> = None;
        let mut n = 0usize;

        for batch in data.iter() {
            let token_ids = batch.token_ids.clone();
            let pos_ids = batch.pos_ids.clone();
            let [batch_size, seq_len] = token_ids.dims();
            let device = token_ids.device();

            let loss_fn = loss_fn.get_or_insert_with(|| loss_config.init::<B>(&device));

            let embeddings = self.forward(token_ids.clone(), pos_ids);

            let noise: Tensor<B, 2> = Tensor::from_data(
                arrgen::normal_array(n as u64, &[embedding_dim, self.sig_reg.slices], 0.0, 1.0)
                    .to_tensor_data()
                    .unwrap(),
                &device,
            );
            let sig_reg_loss = self.sig_reg.forward(embeddings.clone(), noise);

            let logits = self.predictor(embeddings.clone());

            let target_tokens = token_ids
                .slice([0..batch_size, 1..seq_len])
                .reshape([(batch_size * (seq_len - 1))]);
            let vocab_size = logits.dims()[2];
            let pred_logits = logits
                .slice([0..batch_size, 0..(seq_len - 1), 0..vocab_size])
                .reshape([(batch_size * (seq_len - 1)), vocab_size]);

            let batch_loss = loss_fn.forward(pred_logits, target_tokens) + sig_reg_loss;

            let batch_embedding: Tensor<B, 1> =
                embeddings.mean_dim(1).squeeze_dim::<2>(1).mean_dim(0).squeeze_dim::<1>(0);

            n += 1;
            let w = 1.0 / n as f32;
            mean_loss = Some(match mean_loss {
                Some(prev) => prev * (1.0 - w) + batch_loss * w,
                None => batch_loss,
            });
            mean_embedding = Some(match mean_embedding {
                Some(prev) => prev * (1.0 - w) + batch_embedding * w,
                None => batch_embedding,
            });
        }

        let loss = mean_loss.ok_or_else(|| {
            ModelError::FailedTypeVerification("data must contain at least one batch".into())
        })?;
        let embedding = mean_embedding.ok_or_else(|| {
            ModelError::FailedTypeVerification("data must contain at least one batch".into())
        })?;

        Ok(ModelOutput { loss, embedding })
    }
}
