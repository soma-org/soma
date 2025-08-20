//! Embed test ops
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Int, Tensor},
};

#[derive(Module, Debug)]
pub(crate) struct TestEmbed<B: Backend> {
    pub embed: Embedding<B>,
}

impl<B: Backend> TestEmbed<B> {
    pub(crate) fn new(device: &B::Device) -> Self {
        Self {
            embed: EmbeddingConfig::new(1, 10).init(device),
        }
    }
    pub(crate) fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed.forward(input)
    }
}
