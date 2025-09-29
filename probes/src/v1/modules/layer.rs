use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig},
    tensor::{Int, Tensor, backend::Backend},
};

use super::{
    encoder::EncoderConfig,
    pwff::{PositionWiseFeedForward, PositionWiseFeedForwardConfig},
};

#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    norm_1: LayerNorm<B>,
    // attention: MultiHeadAttention<B>,
    norm_2: LayerNorm<B>,
    pwff: PositionWiseFeedForward<B>,
    dropout: Dropout,
}

impl<B: Backend> Layer<B> {
    pub(crate) fn new(config: &EncoderConfig, device: &B::Device) -> Self {
        Layer {
            norm_1: LayerNormConfig::new(config.embedding_dim).init(device),
            norm_2: LayerNormConfig::new(config.embedding_dim).init(device),
            pwff: PositionWiseFeedForwardConfig::new()
                .with_embedding_dim(config.embedding_dim)
                .with_hidden_dim(config.pwff_hidden_dim)
                .with_dropout_rate(config.dropout_rate)
                .with_initializer(config.initializer.clone())
                .init(device),
            dropout: DropoutConfig::new(config.dropout_rate).init(),
        }
    }
    pub(crate) fn forward(
        &self,
        representations: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let x = representations;
        let residual_path = self.norm_1.forward(x.clone());

        // let input_mhs = MhaInput::self_attn(residual_path).mask_attn(mask_attn);
        // let residual_path = self.self_attn.forward(input_mhs).context;

        let residual_path = self.dropout.forward(residual_path);
        let x = x + residual_path;

        let residual_path = self.norm_2.forward(x.clone());
        let residual_path = self.pwff.forward(residual_path);
        let residual_path = self.dropout.forward(residual_path);
        let x = x + residual_path;

        x
    }
}
