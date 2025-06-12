//! MultiHeadAttention test ops
use burn::{
    module::Module,
    nn::attention::{
        generate_autoregressive_mask, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig,
    },
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
        let [batch_size, seq_length, _] = input.dims();
        let device = &self.devices()[0];
        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let input = MhaInput::self_attn(input).mask_attn(mask_attn);
        self.attention.forward(input).context
    }
}
