#![allow(clippy::unwrap_used, clippy::expect_used)]

use arrgen::normal_array;
use burn::backend::NdArray;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Int, PrintOptions, Tensor, Tolerance, set_print_options};
use models::tensor_conversions::ArrayWrapper;
use models::v1::modules::model::ModelConfig;
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
fn test_v1_probe() {
    let seed = 42u64;

    let batch_size = 1usize;
    let vocab_size = 4usize;
    let seq_len = 4usize;
    let num_heads = 2usize;
    let head_dim = 2usize;
    let num_layers = 2;
    let embedding_dim = head_dim * num_heads;
    let hidden_dim = embedding_dim * 2;

    let device = Default::default();
    let mut model = ModelConfig::new()
        .with_embedding_dim(embedding_dim)
        .with_pwff_hidden_dim(hidden_dim)
        .with_num_layers(num_layers)
        .with_num_heads(num_heads)
        .with_vocab_size(vocab_size)
        .init(&device);

    set_print_options(PrintOptions {
        threshold: 1000,    // Default or custom threshold for summarization.
        edge_items: 3,      // Default or custom edge items to display.
        precision: Some(8), // High value for full precision.
    });

    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();

    for l in 0..num_layers {
        let lseed = seed + l as u64;
        tensors.insert(
            format!("encoder.layers.{}.norm_1.gamma", l).to_string(),
            ArrayWrapper(normal_array(lseed + 1, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.norm_1.beta", l).to_string(),
            ArrayWrapper(normal_array(lseed + 2, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.query.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 3, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.query.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 4, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.key.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 5, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.key.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 6, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.value.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 7, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.value.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 8, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.output.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 9, &[embedding_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.output.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 10, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.norm_2.gamma", l).to_string(),
            ArrayWrapper(normal_array(lseed + 11, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.norm_2.beta", l).to_string(),
            ArrayWrapper(normal_array(lseed + 12, &[embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_inner.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 13, &[embedding_dim, hidden_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_inner.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 14, &[hidden_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_outer.weight", l).to_string(),
            ArrayWrapper(normal_array(lseed + 15, &[hidden_dim, embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_outer.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 16, &[embedding_dim], 0.0, 1.0)),
        );
    }
    tensors.insert(
        "final_norm.gamma".to_string(),
        ArrayWrapper(normal_array(seed + 100, &[embedding_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "final_norm.beta".to_string(),
        ArrayWrapper(normal_array(seed + 200, &[embedding_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "embedding.weight".to_string(),
        ArrayWrapper(normal_array(seed + 250, &[vocab_size, embedding_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "predictor.weight".to_string(),
        ArrayWrapper(normal_array(seed + 300, &[embedding_dim, vocab_size], 0.0, 1.0)),
    );
    tensors.insert(
        "predictor.bias".to_string(),
        ArrayWrapper(normal_array(seed + 400, &[vocab_size], 0.0, 1.0)),
    );

    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    model.load_from(&mut store).unwrap();

    let tokens: Tensor<TestBackend, 2, Int> = Tensor::from_ints([[0, 1, 2, 3]], &device);
    let positions: Tensor<TestBackend, 2, Int> =
        Tensor::arange(0..seq_len as i64, &device).unsqueeze().repeat_dim(0, batch_size);
    let output = model.encode(tokens, positions);
    set_print_options(PrintOptions {
        threshold: 1000,    // Default or custom threshold for summarization.
        edge_items: 3,      // Default or custom edge items to display.
        precision: Some(8), // High value for full precision.
    });
    println!("{}", output);

    let expected_output = Tensor::<TestBackend, 3>::from_floats(
        [[
            [2.09210730, 0.69636524, 1.51327145, 2.31296515],
            [1.90634847, 0.86709338, 1.53078938, 2.40285897],
            [1.77797925, 0.66600311, 1.65812206, 2.19417143],
            [1.96223700, 0.77187395, 1.54670000, 2.34615612],
        ]],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
