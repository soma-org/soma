// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used)]

use arrgen::{constant_array, normal_array, uniform_array};
use burn::backend::NdArray;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Int, Tensor, Tolerance};
use burn::{module::Module, prelude::Backend};
use models::tensor_conversions::ArrayWrapper;
use safetensors::serialize;
use std::collections::HashMap;

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

#[derive(Module, Debug)]
pub struct EmbeddingModule<B: Backend> {
    pub embedding: Embedding<B>,
}

impl<B: Backend> EmbeddingModule<B> {
    pub fn new(config: EmbeddingConfig, device: &B::Device) -> Self {
        EmbeddingModule { embedding: config.init(device) }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(input)
    }
}

#[test]
fn test_embedding() {
    let seed = 42u64;
    let batch_size = 1usize;
    let embedding_dim = 2;
    let num_embeddings = 4;
    let config = EmbeddingConfig::new(num_embeddings, embedding_dim);
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "embedding.weight".to_string(),
        ArrayWrapper(normal_array(seed, &[num_embeddings, embedding_dim], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = EmbeddingModule::<TestBackend>::new(config, &device);
    model.load_from(&mut store).unwrap();
    let inputs: Tensor<TestBackend, 2, Int> =
        Tensor::arange(0..num_embeddings as i64, &device).unsqueeze().repeat_dim(0, batch_size);
    let output = model.forward(inputs);
    let expected_output = Tensor::<TestBackend, 3>::from_floats(
        [[
            [0.069427915, 0.13293812],
            [0.26257637, -0.22530088],
            [-0.66422486, -0.2153902],
            [0.19392312, 1.4764173],
        ]],
        &device,
    );
    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
