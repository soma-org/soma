use std::collections::HashMap;

use arrgen::{constant_array, normal_array, uniform_array};
use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::ops::FloatElem;
use burn::tensor::{Int, PrintOptions, Tensor, TensorPrimitive, Tolerance, set_print_options};
use burn_ndarray::NdArrayTensor;
use burn_store::{ModuleSnapshot, SafetensorsStore};
use ndarray_safetensors::TensorViewWithDataBuffer;
use probes::v1::modules::attention::{
    MhaInput, MultiHeadAttention, MultiHeadAttentionConfig, apply_rope,
};
use safetensors::serialize;

type TestBackend = burn_ndarray::NdArray<f32>;
type FT = FloatElem<TestBackend>;

// set_print_options(PrintOptions {
//     threshold: 1000,    // Default or custom threshold for summarization.
//     edge_items: 3,      // Default or custom edge items to display.
//     precision: Some(8), // High value for full precision.
// });
// println!("{}", output);

#[test]
fn test_v1_rope_ones() {
    // inputs: Tensor<B, 4>,         // [batch_size, seq_len, num_heads, head_dim]
    // positions: Tensor<B, 2, Int>, // [batch_size, seq_len]
    let batch_size = 1usize;
    let seq_len = 1usize;
    let num_heads = 1usize;
    let head_dim = 2usize;
    let max_wavelength = 10_000.0;
    let scale_factor = 1.0;
    let device = Default::default();
    let inputs = constant_array(vec![batch_size, seq_len, num_heads, head_dim], 1.0);
    let nd_tensor = NdArrayTensor::from(inputs.into_shared());
    let primitive = TensorPrimitive::Float(nd_tensor);
    let input_tensor: Tensor<TestBackend, 4> = Tensor::from_primitive(primitive);
    let positions: Tensor<TestBackend, 2, Int> = Tensor::from_data([[1]], &device);

    let output = apply_rope(
        input_tensor,
        positions,
        head_dim,
        max_wavelength,
        scale_factor,
    );

    let expected_output =
        Tensor::<TestBackend, 4>::from_floats([[[[-0.30116868, 1.38177323]]]], &device);

    output
        .to_data()
        .assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[derive(Module, Debug)]
pub struct MhaModule<B: Backend> {
    pub mha: MultiHeadAttention<B>,
}

impl<B: Backend> MhaModule<B> {
    pub fn new(head_dim: usize, num_heads: usize, device: &B::Device) -> Self {
        MhaModule {
            mha: MultiHeadAttentionConfig::new(head_dim * num_heads, num_heads)
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
    let seq_len = 1usize;
    let num_heads = 1usize;
    let head_dim = 2usize;
    let max_wavelength = 10_000.0;
    let scale_factor = 1.0;

    let seed = 42u64;
    let mut tensors: HashMap<String, TensorViewWithDataBuffer> = HashMap::new();
    tensors.insert(
        "mha.query.weight".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(
            seed + 1,
            vec![head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.query.bias".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(
            seed + 2,
            vec![head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.key.weight".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(
            seed + 3,
            vec![head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.key.bias".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(
            seed + 4,
            vec![head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.value.weight".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(
            seed + 5,
            vec![head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.value.bias".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(
            seed + 6,
            vec![head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.output.weight".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(
            seed + 7,
            vec![head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.output.bias".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(
            seed + 8,
            vec![head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model: MhaModule<TestBackend> = MhaModule::new(head_dim, num_heads, &device);

    model.apply_from(&mut store).unwrap();

    let inputs = constant_array(vec![batch_size, seq_len, num_heads * head_dim], 1.0);
    let nd_tensor = NdArrayTensor::from(inputs.into_shared());
    let primitive = TensorPrimitive::Float(nd_tensor);
    let input_tensor: Tensor<TestBackend, 3> = Tensor::from_primitive(primitive);
    let positions: Tensor<TestBackend, 2, Int> = Tensor::from_data([[1]], &device);

    let mha_input = MhaInput::new(input_tensor, Some(positions));
    let output = model.forward(mha_input);
    // set_print_options(PrintOptions {
    //     threshold: 1000,    // Default or custom threshold for summarization.
    //     edge_items: 3,      // Default or custom edge items to display.
    //     precision: Some(8), // High value for full precision.
    // });
    // println!("{}", output);

    let expected_output =
        Tensor::<TestBackend, 3>::from_floats([[[-1.93152618, -0.17658120]]], &device);

    output
        .to_data()
        .assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
