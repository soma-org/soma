// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use super::{
    encoder::EncoderConfig,
    pwff::{PositionWiseFeedForward, PositionWiseFeedForwardConfig},
};
use crate::v1::modules::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{Int, Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    norm_1: LayerNorm<B>,
    attention: MultiHeadAttention<B>,
    norm_2: LayerNorm<B>,
    pwff: PositionWiseFeedForward<B>,
}

impl<B: Backend> Layer<B> {
    pub(crate) fn new(config: &EncoderConfig, device: &B::Device) -> Self {
        Layer {
            norm_1: LayerNormConfig::new(config.embedding_dim).init(device),
            attention: MultiHeadAttentionConfig::new()
                .with_num_features(config.embedding_dim)
                .with_num_heads(config.num_heads)
                .with_initializer(config.initializer.clone())
                .init(device),
            norm_2: LayerNormConfig::new(config.embedding_dim).init(device),
            pwff: PositionWiseFeedForwardConfig::new()
                .with_embedding_dim(config.embedding_dim)
                .with_hidden_dim(config.pwff_hidden_dim)
                .with_initializer(config.initializer.clone())
                .init(device),
        }
    }
    pub(crate) fn forward(
        &self,
        context: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let x = context;
        let residual_path = self.norm_1.forward(x.clone());
        let input_mha = MhaInput::new(residual_path, positions);
        let residual_path = self.attention.forward(input_mha);
        let x = x + residual_path;
        let residual_path = self.norm_2.forward(x.clone());
        let residual_path = self.pwff.forward(residual_path);
        x + residual_path
    }
}
