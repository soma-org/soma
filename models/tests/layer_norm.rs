#![allow(clippy::unwrap_used, clippy::expect_used, clippy::useless_vec)]

use arrgen::{constant_array, normal_array, uniform_array};
use burn::backend::NdArray;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Tensor, Tolerance};
use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    prelude::Backend,
};
use models::tensor_conversions::{ArrayWrapper, IntoTensorData};
use safetensors::serialize;
use std::collections::HashMap;

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

// set_print_options(PrintOptions {
//     threshold: 1000,    // Default or custom threshold for summarization.
//     edge_items: 3,      // Default or custom edge items to display.
//     precision: Some(8), // High value for full precision.
// });
// println!("{}", output);

#[test]
fn test_layer_norm_ones() {
    let seed = 42u64;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors
        .insert("layer_norm.gamma".to_string(), ArrayWrapper(normal_array(seed, &[4], 0.0, 1.0)));
    tensors.insert(
        "layer_norm.beta".to_string(),
        ArrayWrapper(normal_array(seed + 1, &[4], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = LayerNormModule::<TestBackend>::new(&device);
    model.load_from(&mut store).unwrap();
    let input_data = constant_array(&vec![4], 1.0).to_tensor_data().unwrap();
    let input_tensor: Tensor<TestBackend, 1> = Tensor::from_data(input_data, &device);
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
    tensors
        .insert("layer_norm.gamma".to_string(), ArrayWrapper(normal_array(seed, &[4], 0.0, 1.0)));
    tensors.insert(
        "layer_norm.beta".to_string(),
        ArrayWrapper(normal_array(seed + 1, &[4], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = LayerNormModule::<TestBackend>::new(&device);
    model.load_from(&mut store).unwrap();

    let input_data = uniform_array(seed + 2, &[4], 0.0, 1.0).to_tensor_data().unwrap();
    let input_tensor: Tensor<TestBackend, 1> = Tensor::from_data(input_data, &device);
    let output = model.forward(input_tensor);
    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [-0.74536324, -2.98460746, -0.31756663, -0.38157958],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
