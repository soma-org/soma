// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::unwrap_used, clippy::expect_used)]

use arrgen::{normal_array_raw, uniform_array_raw};
use burn::Tensor;
use burn::backend::NdArray;
use burn::tensor::ops::FloatElem;
use burn::tensor::{PrintOptions, TensorData, Tolerance, set_print_options};
use models::v1::modules::sig_reg::{SIGReg, SIGRegConfig};

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

#[test]
fn test_v1_sig_reg_normal() {
    let seed = 42;
    let batch_size = 10;
    let seq_len = 10;
    let embedding_dim = 1024;

    let device = Default::default();
    let sig_reg_config = SIGRegConfig::new();
    let sig_reg: SIGReg<TestBackend> = sig_reg_config.init(&device);
    let (v, s) = normal_array_raw(seed + 1, &[batch_size, seq_len, embedding_dim], 0.0, 1.0);
    let input: Tensor<TestBackend, 3> = Tensor::from_data(TensorData::new(v, s), &device);
    let (v, s) = normal_array_raw(seed + 2, &[embedding_dim, sig_reg_config.slices], 0.0, 1.0);
    let noise: Tensor<TestBackend, 2> = Tensor::from_data(TensorData::new(v, s), &device);

    let output = sig_reg.forward(input, noise);
    set_print_options(PrintOptions { threshold: 1000, edge_items: 3, precision: Some(8) });
    println!("{}", output);
    let expected_output = Tensor::<TestBackend, 1>::from_floats([1.28620601], &device);

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[test]
fn test_v1_sig_reg_uniform() {
    let seed = 42;
    let batch_size = 10;
    let seq_len = 10;
    let embedding_dim = 1024;

    let device = Default::default();
    let sig_reg_config = SIGRegConfig::new();
    let sig_reg: SIGReg<TestBackend> = sig_reg_config.init(&device);
    let (v, s) = uniform_array_raw(seed + 1, &[batch_size, seq_len, embedding_dim], 0.0, 1.0);
    let input: Tensor<TestBackend, 3> = Tensor::from_data(TensorData::new(v, s), &device);
    let (v, s) = normal_array_raw(seed + 2, &[embedding_dim, sig_reg_config.slices], 0.0, 1.0);
    let noise: Tensor<TestBackend, 2> = Tensor::from_data(TensorData::new(v, s), &device);

    let output = sig_reg.forward(input, noise);
    set_print_options(PrintOptions { threshold: 1000, edge_items: 3, precision: Some(8) });
    println!("{}", output);
    let expected_output = Tensor::<TestBackend, 1>::from_floats([29.08621025], &device);

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[test]
fn test_v1_sig_reg_small_dim() {
    let seed = 99;
    let batch_size = 2;
    let seq_len = 3;
    let embedding_dim = 8;
    let slices = 4;
    let points = 5;

    let device = Default::default();
    let sig_reg_config = SIGRegConfig::new().with_slices(slices).with_points(points);
    let sig_reg: SIGReg<TestBackend> = sig_reg_config.init(&device);
    let (v, s) = normal_array_raw(seed + 1, &[batch_size, seq_len, embedding_dim], 0.0, 1.0);
    let input: Tensor<TestBackend, 3> = Tensor::from_data(TensorData::new(v, s), &device);
    let (v, s) = normal_array_raw(seed + 2, &[embedding_dim, slices], 0.0, 1.0);
    let noise: Tensor<TestBackend, 2> = Tensor::from_data(TensorData::new(v, s), &device);

    let output = sig_reg.forward(input, noise);
    let expected_output = Tensor::<TestBackend, 1>::from_floats([0.88936800], &device);
    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[test]
fn test_v1_sig_reg_single_batch() {
    let seed = 110;
    let batch_size = 1;
    let seq_len = 1;
    let embedding_dim = 16;

    let device = Default::default();
    let sig_reg_config = SIGRegConfig::new();
    let sig_reg: SIGReg<TestBackend> = sig_reg_config.init(&device);
    let (v, s) = normal_array_raw(seed + 1, &[batch_size, seq_len, embedding_dim], 0.0, 1.0);
    let input: Tensor<TestBackend, 3> = Tensor::from_data(TensorData::new(v, s), &device);
    let (v, s) = normal_array_raw(seed + 2, &[embedding_dim, sig_reg_config.slices], 0.0, 1.0);
    let noise: Tensor<TestBackend, 2> = Tensor::from_data(TensorData::new(v, s), &device);

    let output = sig_reg.forward(input, noise);
    set_print_options(PrintOptions { threshold: 1000, edge_items: 3, precision: Some(8) });
    println!("{}", output);
    let expected_output = Tensor::<TestBackend, 1>::from_floats([3.18785000], &device);
    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
