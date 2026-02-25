// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Int, Tensor};

use crate::v1::{V1_EMBEDDING_DIM, V1_MAX_WAVELENGTH, V1_NUM_HEADS, V1_SCALE_FACTOR};

pub fn apply_rope<B: Backend>(
    inputs: Tensor<B, 4>,         // [batch_size, seq_len, num_heads, head_dim]
    positions: Tensor<B, 2, Int>, // [batch_size, seq_len]
    head_dim: usize,
    max_wavelength: f32,
    scale_factor: f32,
) -> Tensor<B, 4> {
    if scale_factor < 1.0 {
        panic!("scale_factor must be >= 1.0, got {}", scale_factor);
    }
    if !head_dim.is_multiple_of(2) {
        panic!("head_dim must be even, got {}", head_dim);
    }

    // Calculate frequencies: 1 / (max_wavelength ^ (2i / head_dim))
    let device = inputs.device();

    let fraction = Tensor::<B, 1, Int>::arange(0..(head_dim / 2) as i64, &device)
        .float()
        .mul_scalar(2)
        .div_scalar(head_dim as i64);
    let base_tensor = Tensor::full_like(&fraction, max_wavelength);
    let timescale = base_tensor.powf(fraction);

    // Prepare sinusoid input: positions * theta
    let positions_float = positions.float().unsqueeze_dim::<3>(2); // [batch_size, seq_len] -> [batch_size, seq_len, 1]
    let theta = timescale.unsqueeze::<2>().unsqueeze::<3>(); // [head_dim/2] -> [1, head_dim/2] -> [1, 1, head_dim/2]
    let sinusoid_inp = positions_float.div(theta);
    let sinusoid_inp = sinusoid_inp.unsqueeze_dim::<4>(2);
    let sinusoid_inp = sinusoid_inp.div_scalar(scale_factor);
    let sin = sinusoid_inp.clone().sin();
    let cos = sinusoid_inp.cos();

    let chunks = inputs.chunk(2, 3);
    if chunks.len() != 2 {
        panic!("Expected 2 chunks from splitting head_dim, got {}", chunks.len());
    }
    let first_half = chunks[0].clone();
    let second_half = chunks[1].clone();

    let first_part = first_half.clone() * cos.clone() - second_half.clone() * sin.clone();
    let second_part = second_half * cos + first_half * sin;

    Tensor::cat(vec![first_part, second_part], 3)
}

/// Configuration to create a [Multi Head Attention](MultiHeadAttention) layer using the [init function](MultiHeadAttentionConfig::init).
#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    /// Feature size (same size for input, keys, query, out, etc.)
    #[config(default = "V1_EMBEDDING_DIM")]
    pub num_features: usize,
    /// The number of heads.
    #[config(default = "V1_NUM_HEADS")]
    pub num_heads: usize,
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
    /// The max wavelength for RoPE.
    #[config(default = "V1_MAX_WAVELENGTH")]
    pub max_wavelength: f32,
    /// The RoPE scale factor.
    #[config(default = "V1_SCALE_FACTOR")]
    pub scale_factor: f32,
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
    /// The size of each linear layer.
    pub num_features: usize,
    /// The number of heads.
    pub num_heads: usize,
    /// The dimension per head.
    pub head_dim: usize,
    /// Minimum value a float can take.
    pub min_float: f64,
    pub max_wavelength: f32,
    pub scale_factor: f32,
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
            num_features: self.num_features,
            num_heads: self.num_heads,
            head_dim: self.num_features / self.num_heads,
            min_float: self.min_float,
            max_wavelength: self.max_wavelength,
            scale_factor: self.scale_factor,
        }
    }
}

/// [Multihead attention](MultiHeadAttention) forward pass input argument.
#[derive(Debug, Clone)]
pub struct MhaInput<B: Backend> {
    /// Shape `[batch_size, seq_length, num_features]`
    query: Tensor<B, 3>,
    /// Shape `[batch_size, seq_length]`
    positions: Option<Tensor<B, 2, Int>>,
    mask_pad: Option<Tensor<B, 2, Bool>>,
    mask_attn: Option<Tensor<B, 3, Bool>>,
}

impl<B: Backend> MhaInput<B> {
    pub fn new(query: Tensor<B, 3>, positions: Tensor<B, 2, Int>) -> Self {
        let [batch_size, seq_length, _d_model] = query.dims();

        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, &query.device());
        Self { query, positions: Some(positions), mask_attn: Some(mask_attn), mask_pad: None }
    }
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(&self, input: MhaInput<B>) -> Tensor<B, 3> {
        let [batch_size, seq_length, d_model] = input.query.dims();

        let mut query = self.attention_linear(input.query.clone(), &self.query);
        let mut key = self.attention_linear(input.query.clone(), &self.key);
        let value = self.attention_linear(input.query, &self.value);

        if let Some(positions) = input.positions {
            query = query.swap_dims(1, 2);
            key = key.swap_dims(1, 2);

            query = apply_rope(
                query,
                positions.clone(),
                self.head_dim,
                self.max_wavelength,
                self.scale_factor,
            );
            key = apply_rope(key, positions, self.head_dim, self.max_wavelength, self.scale_factor);

            query = query.swap_dims(1, 2);
            key = key.swap_dims(1, 2);
        }

        let attn_scores = self.attn_scores(query, key);
        let weights = self.attn_weights(attn_scores, input.mask_pad, input.mask_attn);
        let context = weights.clone().matmul(value);
        let context = context.swap_dims(1, 2).reshape([batch_size, seq_length, d_model]);
        self.output.forward(context)
    }

    fn attn_scores(&self, query: Tensor<B, 4>, key: Tensor<B, 4>) -> Tensor<B, 4> {
        query.matmul(key.transpose()).div_scalar((self.head_dim as f32).sqrt())
    }

    fn attn_weights(
        &self,
        mut attn_scores: Tensor<B, 4>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
        mask_attn: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 4> {
        if let Some(mask_pad) = mask_pad {
            let [batch_size, seq_length] = mask_pad.dims();

            attn_scores = attn_scores
                .mask_fill(mask_pad.reshape([batch_size, 1, 1, seq_length]), self.min_float);
        }

        if let Some(mask_attn) = mask_attn {
            let [batch_size, seq_length_1, seq_length_2] = mask_attn.dims();

            attn_scores = attn_scores.mask_fill(
                mask_attn.reshape([batch_size, 1, seq_length_1, seq_length_2]),
                self.min_float,
            );
        }

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
