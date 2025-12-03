use crate::v1::{V1_EMBEDDING_DIM, V1_NUM_HEADS};
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Int, Tensor, activation::softmax},
};

/// Configuration to create a [Multi Head Attention](MultiHeadAttention) layer using the [init function](MultiHeadAttentionConfig::init).
#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    /// Feature size (same size for input, keys, query, out, etc.)
    #[config(default = "V1_EMBEDDING_DIM")]
    pub num_features: usize,
    /// The number of heads.
    #[config(default = "V1_NUM_HEADS")]
    pub num_heads: usize,
    /// The probability that dropout occurs
    #[config(default = 0.0)]
    pub dropout_rate: f64,
    /// The minimum value a float can take. Default: -1.0e4
    /// This is used to mask attention scores before calculating attention weights.
    /// A value too low might result in NaN.
    #[config(default = -1.0e4)]
    pub min_float: f64,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Linear layer to transform the input features into the query space.
    pub query: Linear<B>,
    /// Linear layer to transform the input features into the key space.
    pub key: Linear<B>,
    /// Linear layer to transform the input features into the value space.
    pub value: Linear<B>,
    /// Linear layer to transform the output features back to the original space.
    pub output: Linear<B>,
    /// Dropout layer.
    pub dropout: Dropout,
    /// The size of each linear layer.
    pub num_features: usize,
    /// The number of heads.
    pub num_heads: usize,
    /// The dimension per head.
    pub head_dim: usize,
    /// Minimum value a float can take.
    pub min_float: f64,
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let linear = |config: &Self| {
            LinearConfig::new(config.num_features, config.num_features)
                .with_initializer(self.initializer.clone())
                .init(device)
        };

        MultiHeadAttention {
            query: linear(self),
            key: linear(self),
            value: linear(self),
            output: linear(self),
            dropout: DropoutConfig::new(self.dropout_rate).init(),
            num_features: self.num_features,
            num_heads: self.num_heads,
            head_dim: self.num_features / self.num_heads,
            min_float: self.min_float,
        }
    }
}

/// [Multihead attention](MultiHeadAttention) forward pass input argument.
#[derive(Debug, Clone)]
pub struct MhaInput<B: Backend> {
    /// Shape `[batch_size, seq_length, num_features]`
    query: Tensor<B, 3>,
}

impl<B: Backend> MhaInput<B> {
    pub fn new(query: Tensor<B, 3>) -> Self {
        Self { query }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(&self, input: MhaInput<B>) -> Tensor<B, 3> {
        let [batch_size, seq_length_1, d_model] = input.query.dims();

        let query = self.attention_linear(input.query.clone(), &self.query);
        let key = self.attention_linear(input.query.clone(), &self.key);
        let value = self.attention_linear(input.query, &self.value);

        let attn_scores = self.attn_scores(query, key);
        let weights = self.attn_weights(attn_scores);

        let context = weights.clone().matmul(value);
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length_1, d_model]);
        let context = self.output.forward(context);

        context
    }

    fn attn_scores(&self, query: Tensor<B, 4>, key: Tensor<B, 4>) -> Tensor<B, 4> {
        let attn_scores = query
            .matmul(key.transpose())
            .div_scalar((self.head_dim as f32).sqrt());

        self.dropout.forward(attn_scores)
    }

    fn attn_weights(&self, attn_scores: Tensor<B, 4>) -> Tensor<B, 4> {
        softmax(attn_scores, 3)
    }

    fn attention_linear(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.num_heads, self.head_dim])
            .swap_dims(1, 2)
    }
}
