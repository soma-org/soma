use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Int, Tensor, backend::Backend},
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
    /// The vocab size.
    #[config(default = "V1_VOCAB_SIZE")]
    pub vocab_size: usize,
    /// The probability that dropout occurs
    #[config(default = 0.0)]
    pub dropout_rate: f64,
    /// The max wavelength for RoPE.
    #[config(default = "V1_MAX_WAVELENGTH")]
    pub max_wavelength: f32,
    /// The RoPE scale factor.
    #[config(default = "V1_SCALE_FACTOR")]
    pub scale_factor: f32,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub weight_initializer: Initializer,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::Normal{mean:0.0, std:1.0}")]
    pub token_initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct Probe<B: Backend> {
    mask_token: Param<Tensor<B, 3>>,
    encoder: Encoder<B>,
    final_norm: LayerNorm<B>,
    predictor: Linear<B>,
}

impl ProbeConfig {
    /// Initialize a new module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Probe<B> {
        Probe {
            mask_token: self
                .token_initializer
                .init([1, 1, self.embedding_dim], device),
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
    pub fn forward(&self, context: Tensor<B, 3>, positions: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let batch_size = context.shape().dims[0];
        let mask_token = self
            .mask_token
            .val()
            .repeat(&vec![batch_size, 1, 1])
            .require_grad();
        let x = Tensor::cat(vec![mask_token, context], 1);
        let positions = Tensor::cat(
            vec![
                Tensor::zeros([batch_size, 1], &positions.device()),
                positions,
            ],
            1,
        );

        let x = self.encoder.forward(x, positions);
        let x = self.final_norm.forward(x);

        let mask_token = x.slice([0..batch_size, 0..1]).squeeze_dim(1);
        self.predictor.forward(mask_token)
    }
}
