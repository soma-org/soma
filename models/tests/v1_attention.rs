#![allow(clippy::unwrap_used, clippy::expect_used, clippy::useless_vec)]

use std::collections::HashMap;

use arrgen::{constant_array, normal_array, uniform_array};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayTensor;
use burn::module::Module;

use burn::prelude::Backend;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Int, PrintOptions, Tensor, TensorPrimitive, Tolerance, set_print_options};
use models::tensor_conversions::{ArrayWrapper, IntoTensorData};
use models::v1::modules::attention::{
    MhaInput, MultiHeadAttention, MultiHeadAttentionConfig, apply_rope,
};
use safetensors::serialize;

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

// set_print_options(PrintOptions {
//     threshold: 1000,    // Default or custom threshold for summarization.
//     edge_items: 3,      // Default or custom edge items to display.
//     precision: Some(8), // High value for full precision.
// });
// println!("{}", output);

#[test]
fn test_v1_rope_ones() {
    let batch_size = 1usize;
    let seq_len = 1usize;
    let num_heads = 1usize;
    let head_dim = 2usize;
    let max_wavelength = 10_000.0;
    let scale_factor = 1.0;
    let device = Default::default();
    let inputs = constant_array(&vec![batch_size, seq_len, num_heads, head_dim], 1.0);
    let nd_tensor = NdArrayTensor::from(inputs.into_shared());
    let primitive = TensorPrimitive::Float(nd_tensor);
    let input_tensor: Tensor<TestBackend, 4> = Tensor::from_primitive(primitive);
    let positions: Tensor<TestBackend, 2, Int> = Tensor::from_data([[1]], &device);

    let output = apply_rope(input_tensor, positions, head_dim, max_wavelength, scale_factor);

    let expected_output =
        Tensor::<TestBackend, 4>::from_floats([[[[-0.30116868, 1.38177323]]]], &device);

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[derive(Module, Debug)]
pub struct MhaModule<B: Backend> {
    pub mha: MultiHeadAttention<B>,
}

impl<B: Backend> MhaModule<B> {
    pub fn new(head_dim: usize, num_heads: usize, device: &B::Device) -> Self {
        MhaModule {
            mha: MultiHeadAttentionConfig::new()
                .with_num_features(head_dim * num_heads)
                .with_num_heads(num_heads)
                .with_dropout_rate(0.0)
                .init(device),
        }
    }

    pub fn forward(&self, input: MhaInput<B>) -> Tensor<B, 3> {
        self.mha.forward(input)
    }
}

#[test]
fn test_v1_attention() {
    let batch_size = 1usize;
    let seq_len = 4usize;
    let num_heads = 2usize;
    let head_dim = 2usize;

    let seed = 42u64;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "mha.query.weight".to_string(),
        ArrayWrapper(normal_array(
            seed + 1,
           &[head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.query.bias".to_string(),
        ArrayWrapper(normal_array(seed + 2,&[head_dim * num_heads], 0.0, 1.0)),
    );
    tensors.insert(
        "mha.key.weight".to_string(),
        ArrayWrapper(normal_array(
            seed + 3,
           &[head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.key.bias".to_string(),
        ArrayWrapper(normal_array(seed + 4,&[head_dim * num_heads], 0.0, 1.0)),
    );
    tensors.insert(
        "mha.value.weight".to_string(),
        ArrayWrapper(normal_array(
            seed + 5,
           &[head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.value.bias".to_string(),
        ArrayWrapper(normal_array(seed + 6,&[head_dim * num_heads], 0.0, 1.0)),
    );
    tensors.insert(
        "mha.output.weight".to_string(),
        ArrayWrapper(normal_array(
            seed + 7,
           &[head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.output.bias".to_string(),
        ArrayWrapper(normal_array(seed + 8,&[head_dim * num_heads], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model: MhaModule<TestBackend> = MhaModule::new(head_dim, num_heads, &device);

    model.load_from(&mut store).unwrap();

    let input_data =
        normal_array(seed + 9,&[batch_size, seq_len, num_heads * head_dim], 0.0, 1.0)
            .to_tensor_data()
            .unwrap();
    let input_tensor: Tensor<TestBackend, 3> = Tensor::from_data(input_data, &device);
    let positions: Tensor<TestBackend, 2, Int> =
        Tensor::arange(0..seq_len as i64, &device).unsqueeze().repeat_dim(0, batch_size);
    println!("{}", positions);
    let mha_input = MhaInput::new(input_tensor, positions);
    let output = model.forward(mha_input);
    set_print_options(PrintOptions {
        threshold: 1000,    // Default or custom threshold for summarization.
        edge_items: 3,      // Default or custom edge items to display.
        precision: Some(8), // High value for full precision.
    });

    let expected_output = Tensor::<TestBackend, 3>::from_floats(
        [[
            [-3.59691596, -0.79137892, 2.59000635, 1.15417659],
            [-1.89609909, 0.90684074, 4.74483156, 9.88230705],
            [4.25082970, 2.23179317, 2.04289985, 9.17799950],
            [1.67264295, 1.73140073, 3.30692792, 10.11232185],
        ]],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
