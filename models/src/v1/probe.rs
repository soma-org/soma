use burn::{
    config::Config,
    module::Module,
    nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Bool, Int, Tensor, backend::Backend},
};

use super::{
    V1_EMBEDDING_DIM, V1_MAX_WAVELENGTH, V1_NUM_HEADS, V1_NUM_LAYERS, V1_PWFF_HIDDEN_DIM,
    V1_SCALE_FACTOR, V1_VOCAB_SIZE,
    modules::encoder::{Encoder, EncoderConfig},
};

#[derive(Config, Debug)]
pub struct ProbeConfig {
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
}

#[derive(Module, Debug)]
pub struct Probe<B: Backend> {
    encoder: Encoder<B>,
    final_norm: LayerNorm<B>,
    predictor: Linear<B>,
}

impl ProbeConfig {
    /// Initialize a new module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Probe<B> {
        Probe {
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
        }
    }
}

impl<B: Backend> Probe<B> {
    pub fn forward(
        &self,
        context: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        attn_mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        let x = self.encoder.forward(context, positions, attn_mask);
        let x = self.final_norm.forward(x);
        x
    }

    pub fn predictor(&self, embeddings: Tensor<B, 3>) -> Tensor<B, 3> {
        self.predictor.forward(embeddings)
    }
}
