//! Linear test ops
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub(crate) struct TestLinear<B: Backend> {
    pub linear: Linear<B>,
}

impl<B: Backend> TestLinear<B> {
    pub(crate) fn new(device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(10, 10).init(device),
        }
    }
    pub(crate) fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        self.linear.forward(input)
    }
}
