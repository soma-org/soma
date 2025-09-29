use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Gelu, Initializer, Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

use crate::v1::{V1_EMBEDDING_DIM, V1_PWFF_HIDDEN_DIM};

#[derive(Config, Debug)]
pub struct PositionWiseFeedForwardConfig {
    /// The size of the input and output features.
    #[config(default = "V1_EMBEDDING_DIM")]
    pub embedding_dim: usize,
    /// The size of the hidden inner features.
    #[config(default = "V1_PWFF_HIDDEN_DIM")]
    pub hidden_dim: usize,
    /// The probability that dropout occurs
    #[config(default = 0.0)]
    pub dropout_rate: f64,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    pub linear_inner: Linear<B>,
    pub linear_outer: Linear<B>,
    pub dropout: Dropout,
    pub gelu: Gelu,
}

impl PositionWiseFeedForwardConfig {
    /// Initialize a new [position-wise feed-forward](PositionWiseFeedForward) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionWiseFeedForward<B> {
        PositionWiseFeedForward {
            linear_inner: LinearConfig::new(self.embedding_dim, self.hidden_dim)
                .with_initializer(self.initializer.clone())
                .init(device),
            linear_outer: LinearConfig::new(self.hidden_dim, self.embedding_dim)
                .with_initializer(self.initializer.clone())
                .init(device),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            gelu: Gelu::new(),
        }
    }
}

impl<B: Backend> PositionWiseFeedForward<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);
        self.linear_outer.forward(x)
    }
}
