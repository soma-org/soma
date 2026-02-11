#![allow(clippy::unwrap_used, clippy::expect_used)]

use arrgen::{constant_array, normal_array};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayTensor;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Int, PrintOptions, Tensor, TensorPrimitive, Tolerance, set_print_options};
use models::tensor_conversions::{ArrayWrapper, IntoTensorData};
use models::v1::modules::encoder::EncoderConfig;
use safetensors::serialize;
use std::collections::HashMap;

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

// set_print_options(PrintOptions {
//     threshold: 1000,    // Default or custom threshold for summarization.
//     edge_items: 3,      // Default or custom edge items to display.
//     precision: Some(8), // High value for full precision.
// });
// println!("{}", output);

#[test]
fn test_v1_encoder_ones() {
    let seed = 42u64;
    let batch_size = 1usize;
    let seq_len = 4usize;
    let num_heads = 2usize;
    let head_dim = 2usize;
    let num_layers = 2usize;
    let embedding_dim = head_dim * num_heads;
    let hidden_dim = embedding_dim * 2;

    let device = Default::default();
    let mut model = EncoderConfig::new()
        .with_embedding_dim(embedding_dim)
        .with_pwff_hidden_dim(hidden_dim)
        .with_num_layers(num_layers)
        .with_num_heads(num_heads)
        .with_dropout_rate(0.0)
        .init(&device);

    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();

    for l in 0..num_layers {
        let lseed = seed + l as u64;
        tensors.insert(
            format!("layers.{}.norm_1.gamma", l).to_string(),
            ArrayWrapper(normal_array(lseed + 1, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.norm_1.beta", l).to_string(),
            ArrayWrapper(normal_array(lseed + 2, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.query.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 3, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.query.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 4, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.key.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 5, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.key.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 6, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.value.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 7, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.value.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 8, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.output.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 9, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.output.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 10, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.norm_2.gamma", l).to_string(),
            ArrayWrapper(normal_array(lseed + 11, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.norm_2.beta", l).to_string(),
            ArrayWrapper(normal_array(lseed + 12, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.pwff.linear_inner.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 13, &[embedding_dim, hidden_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.pwff.linear_inner.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 14, &[hidden_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.pwff.linear_outer.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 15, &[hidden_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.pwff.linear_outer.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 16, &[embedding_dim], 0.0, 1.0)),
        );
    }

    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    model.load_from(&mut store).unwrap();

    let input_data = normal_array(seed + 100, &[batch_size, seq_len, embedding_dim], 0.0, 1.0)
        .to_tensor_data()
        .unwrap();
    let input_tensor: Tensor<TestBackend, 3> = Tensor::from_data(input_data, &device);
    let positions: Tensor<TestBackend, 2, Int> =
        Tensor::arange(0..seq_len as i64, &device).unsqueeze().repeat_dim(0, batch_size);
    let output = model.forward(input_tensor, positions);

    let expected_output = Tensor::<TestBackend, 3>::from_floats(
        [[
            [8.33097267, 3.23748636, 12.93889809, -4.79724693],
            [5.95732164, 2.65965629, 11.97813892, -3.51363301],
            [8.32810211, 4.93732452, 12.48622322, -4.29308748],
            [3.45333171, 3.12663937, 10.79409981, -2.80449820],
        ]],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
