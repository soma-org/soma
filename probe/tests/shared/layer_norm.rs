//! LayerNorm test ops
use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub(crate) struct TestLayerNorm<B: Backend> {
    pub layer_norm: LayerNorm<B>,
}

impl<B: Backend> TestLayerNorm<B> {
    pub(crate) fn new(device: &B::Device) -> Self {
        Self {
            layer_norm: LayerNormConfig::new(10).init(device),
        }
    }
    pub(crate) fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        self.layer_norm.forward(input)
    }
}
