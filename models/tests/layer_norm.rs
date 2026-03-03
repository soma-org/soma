// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::useless_vec)]

use std::collections::HashMap;

use arrgen::{constant_array_raw, normal_array_raw, uniform_array_raw};
use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::Backend;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Tensor, TensorData, Tolerance};
use models::tensor_conversions::ArrayWrapper;
use safetensors::serialize;

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

#[derive(Module, Debug)]
pub struct LayerNormModule<B: Backend> {
    pub layer_norm: LayerNorm<B>,
}

impl<B: Backend> LayerNormModule<B> {
    pub fn new(device: &B::Device) -> Self {
        LayerNormModule { layer_norm: LayerNormConfig::new(4).init(device) }
    }

    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        self.layer_norm.forward(input)
    }
}

#[test]
fn test_layer_norm_ones() {
    let seed = 42u64;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert("layer_norm.gamma".to_string(), normal_array_raw(seed, &[4], 0.0, 1.0).into());
    tensors
        .insert("layer_norm.beta".to_string(), normal_array_raw(seed + 1, &[4], 0.0, 1.0).into());
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = LayerNormModule::<TestBackend>::new(&device);
    model.load_from(&mut store).unwrap();
    let (v, s) = constant_array_raw(&vec![4], 1.0);
    let input_tensor: Tensor<TestBackend, 1> = Tensor::from_data(TensorData::new(v, s), &device);
    let output = model.forward(input_tensor);
    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [0.26803425, -0.30034754, -0.18579677, -0.37248048],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[test]
fn test_layer_norm_uniform() {
    let seed = 44u64;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert("layer_norm.gamma".to_string(), normal_array_raw(seed, &[4], 0.0, 1.0).into());
    tensors
        .insert("layer_norm.beta".to_string(), normal_array_raw(seed + 1, &[4], 0.0, 1.0).into());
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = LayerNormModule::<TestBackend>::new(&device);
    model.load_from(&mut store).unwrap();

    let (v, s) = uniform_array_raw(seed + 2, &[4], 0.0, 1.0);
    let input_tensor: Tensor<TestBackend, 1> = Tensor::from_data(TensorData::new(v, s), &device);
    let output = model.forward(input_tensor);
    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [-0.74536324, -2.98460746, -0.31756663, -0.38157958],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
