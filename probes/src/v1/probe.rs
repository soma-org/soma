use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

use super::{
    V1_EMBEDDING_DIM, V1_NUM_LAYERS, V1_PWFF_HIDDEN_DIM, V1_VOCAB_SIZE,
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
    /// The vocab size.
    #[config(default = "V1_VOCAB_SIZE")]
    pub vocab_size: usize,
    /// The probability that dropout occurs
    #[config(default = 0.0)]
    pub dropout_rate: f64,
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
    mask_token: Param<Tensor<B, 1>>,
    encoder: Encoder<B>,
    final_norm: LayerNorm<B>,
    predictor: Linear<B>,
}

impl ProbeConfig {
    /// Initialize a new module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Probe<B> {
        Probe {
            mask_token: self.token_initializer.init([self.embedding_dim], device),
            encoder: EncoderConfig::new()
                .with_embedding_dim(self.embedding_dim)
                .with_pwff_hidden_dim(self.pwff_hidden_dim)
                .with_num_layers(self.num_layers)
                .with_dropout_rate(self.dropout_rate)
                .with_initializer(self.weight_initializer.clone())
                .init(device),
            final_norm: LayerNormConfig::new(self.embedding_dim).init(device),
            predictor: LinearConfig::new(self.embedding_dim, self.vocab_size)
                .with_initializer(self.weight_initializer.clone())
                .init(device),
        }
    }
}

// impl<B: Backend> Probe<B> {
//     pub fn forward(
//         &self,
//         target_byte_index: Tensor<B, 1, Int>,
//         context_embeddings: Tensor<B, 3>,
//         context_byte_indices: Tensor<B, 2, Int>,
//     ) -> Tensor<B, 3> {
//     }
// }
