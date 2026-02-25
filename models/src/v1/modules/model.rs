// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::v1::modules::encoder::{Encoder, EncoderConfig};
use crate::v1::{
    V1_EMBEDDING_DIM, V1_MAX_WAVELENGTH, V1_NUM_HEADS, V1_NUM_LAYERS, V1_PWFF_HIDDEN_DIM,
    V1_SCALE_FACTOR, V1_VOCAB_SIZE,
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
pub struct Model<B: Backend> {
    embedding: Embedding<B>,
    encoder: Encoder<B>,
    final_norm: LayerNorm<B>,
    predictor: Linear<B>,
}

impl ModelConfig {
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

impl<B: Backend> Model<B> {
    pub fn encode(&self, tokens: Tensor<B, 2, Int>, positions: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(tokens);
        let x = self.encoder.forward(x, positions);
        self.final_norm.forward(x)
    }

    pub fn predict(&self, embeddings: Tensor<B, 3>) -> Tensor<B, 3> {
        self.predictor.forward(embeddings)
    }
}
