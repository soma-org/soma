//! MultiHeadAttention test ops
use burn::{
    module::Module,
    nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub(crate) struct TestMultiHeadAttention<B: Backend> {
    pub attention: MultiHeadAttention<B>,
}

impl<B: Backend> TestMultiHeadAttention<B> {
    pub(crate) fn new(device: &B::Device) -> Self {
        Self {
            attention: MultiHeadAttentionConfig::new(10, 2)
                .with_dropout(0.0)
                .init(device),
        }
    }
    pub(crate) fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let input = MhaInput::self_attn(input);
        self.attention.forward(input).context
    }
}
