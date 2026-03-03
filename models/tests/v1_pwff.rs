// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;

use arrgen::{constant_array_raw, normal_array_raw};
use burn::backend::NdArray;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Tensor, TensorData, Tolerance};
use models::tensor_conversions::ArrayWrapper;
use models::v1::modules::pwff::PositionWiseFeedForwardConfig;
use safetensors::serialize;

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

#[test]
fn test_v1_pwff_ones() {
    let embedding_dim = 4usize;
    let hidden_dim = 2usize;
    let seed = 42u64;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "linear_inner.weight".to_string(),
        normal_array_raw(seed + 1, &[embedding_dim, hidden_dim], 0.0, 1.0).into(),
    );
    tensors.insert(
        "linear_inner.bias".to_string(),
        normal_array_raw(seed + 2, &[hidden_dim], 0.0, 1.0).into(),
    );
    tensors.insert(
        "linear_outer.weight".to_string(),
        normal_array_raw(seed + 3, &[hidden_dim, embedding_dim], 0.0, 1.0).into(),
    );
    tensors.insert(
        "linear_outer.bias".to_string(),
        normal_array_raw(seed + 4, &[embedding_dim], 0.0, 1.0).into(),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = PositionWiseFeedForwardConfig::new()
        .with_embedding_dim(embedding_dim)
        .with_hidden_dim(hidden_dim)
        .init(&device);
    model.load_from(&mut store).unwrap();
    let (v, s) = constant_array_raw(&[embedding_dim], 1.0);
    let input_tensor: Tensor<TestBackend, 1> = Tensor::from_data(TensorData::new(v, s), &device);
    let output = model.forward(input_tensor);

    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [0.31442693, 0.70205802, -2.13397980, -1.71679294],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[test]
fn test_v1_pwff_normal_input() {
    let embedding_dim = 4usize;
    let hidden_dim = 2usize;
    let seed = 50u64;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "linear_inner.weight".to_string(),
        normal_array_raw(seed + 1, &[embedding_dim, hidden_dim], 0.0, 1.0).into(),
    );
    tensors.insert(
        "linear_inner.bias".to_string(),
        normal_array_raw(seed + 2, &[hidden_dim], 0.0, 1.0).into(),
    );
    tensors.insert(
        "linear_outer.weight".to_string(),
        normal_array_raw(seed + 3, &[hidden_dim, embedding_dim], 0.0, 1.0).into(),
    );
    tensors.insert(
        "linear_outer.bias".to_string(),
        normal_array_raw(seed + 4, &[embedding_dim], 0.0, 1.0).into(),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = PositionWiseFeedForwardConfig::new()
        .with_embedding_dim(embedding_dim)
        .with_hidden_dim(hidden_dim)
        .init(&device);
    model.load_from(&mut store).unwrap();
    let (v, s) = normal_array_raw(seed + 5, &[embedding_dim], 0.0, 1.0);
    let input_tensor: Tensor<TestBackend, 1> = Tensor::from_data(TensorData::new(v, s), &device);
    let output = model.forward(input_tensor);
    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [-0.42594010, -0.70958626, -0.26518542, -0.35035765],
        &device,
    );
    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[test]
fn test_v1_pwff_batched() {
    let batch_size = 2usize;
    let seq_len = 3usize;
    let embedding_dim = 4usize;
    let hidden_dim = 2usize;
    let seed = 60u64;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "linear_inner.weight".to_string(),
        normal_array_raw(seed + 1, &[embedding_dim, hidden_dim], 0.0, 1.0).into(),
    );
    tensors.insert(
        "linear_inner.bias".to_string(),
        normal_array_raw(seed + 2, &[hidden_dim], 0.0, 1.0).into(),
    );
    tensors.insert(
        "linear_outer.weight".to_string(),
        normal_array_raw(seed + 3, &[hidden_dim, embedding_dim], 0.0, 1.0).into(),
    );
    tensors.insert(
        "linear_outer.bias".to_string(),
        normal_array_raw(seed + 4, &[embedding_dim], 0.0, 1.0).into(),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = PositionWiseFeedForwardConfig::new()
        .with_embedding_dim(embedding_dim)
        .with_hidden_dim(hidden_dim)
        .init(&device);
    model.load_from(&mut store).unwrap();
    let (v, s) = normal_array_raw(seed + 5, &[batch_size, seq_len, embedding_dim], 0.0, 1.0);
    let input_tensor: Tensor<TestBackend, 3> = Tensor::from_data(TensorData::new(v, s), &device);
    let output = model.forward(input_tensor);
    let expected_output = Tensor::<TestBackend, 3>::from_floats(
        [
            [
                [2.99256825, -2.32797265, -3.22312760, 1.21508217],
                [2.70691204, -2.19828272, -3.50960517, 0.83996159],
                [-0.04863004, 0.30158886, -1.44390535, 1.00410569],
            ],
            [
                [0.16999049, 0.10190730, -1.61299801, 0.98700720],
                [0.09025692, 0.22779667, -1.34613645, 1.15397000],
                [0.28953564, -0.03523720, -1.81356251, 0.89298093],
            ],
        ],
        &device,
    );
    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
