#![allow(clippy::unwrap_used, clippy::expect_used)]

use arrgen::{normal_array, uniform_array};
use burn::tensor::ops::FloatElem;
use burn::tensor::{PrintOptions, Tolerance, set_print_options};
use burn::{Tensor, backend::NdArray};
use models::tensor_conversions::IntoTensorData;
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
    let input: Tensor<TestBackend, 3> = Tensor::from_data(
        normal_array(seed + 1, &[batch_size, seq_len, embedding_dim], 0.0, 1.0)
            .to_tensor_data()
            .unwrap(),
        &device,
    );
    let noise: Tensor<TestBackend, 2> = Tensor::from_data(
        normal_array(seed + 2, &[embedding_dim, sig_reg_config.slices], 0.0, 1.0)
            .to_tensor_data()
            .unwrap(),
        &device,
    );

    let output = sig_reg.forward(input, noise);
    set_print_options(PrintOptions {
        threshold: 1000,    // Default or custom threshold for summarization.
        edge_items: 3,      // Default or custom edge items to display.
        precision: Some(8), // High value for full precision.
    });
    println!("{}", output);
    let expected_output = Tensor::<TestBackend, 1>::from_floats([1.33955204], &device);

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
    let input: Tensor<TestBackend, 3> = Tensor::from_data(
        uniform_array(seed + 1, &[batch_size, seq_len, embedding_dim], 0.0, 1.0)
            .to_tensor_data()
            .unwrap(),
        &device,
    );
    let noise: Tensor<TestBackend, 2> = Tensor::from_data(
        normal_array(seed + 2, &[embedding_dim, sig_reg_config.slices], 0.0, 1.0)
            .to_tensor_data()
            .unwrap(),
        &device,
    );

    let output = sig_reg.forward(input, noise);
    set_print_options(PrintOptions {
        threshold: 1000,    // Default or custom threshold for summarization.
        edge_items: 3,      // Default or custom edge items to display.
        precision: Some(8), // High value for full precision.
    });
    println!("{}", output);
    let expected_output = Tensor::<TestBackend, 1>::from_floats([121.57125092], &device);

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
