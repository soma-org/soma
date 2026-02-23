#![allow(clippy::unwrap_used, clippy::expect_used, clippy::useless_vec)]

use arrgen::{constant_array, normal_array, uniform_array};
use burn::backend::NdArray;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Tensor, Tolerance};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
};
use models::tensor_conversions::{ArrayWrapper, IntoTensorData};
use safetensors::serialize;
use std::collections::HashMap;

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

#[derive(Module, Debug)]
pub struct LinearModule<B: Backend> {
    pub linear: Linear<B>,
}

impl<B: Backend> LinearModule<B> {
    pub fn new(device: &B::Device) -> Self {
        LinearModule { linear: LinearConfig::new(2, 4).init(device) }
    }

    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        self.linear.forward(input)
    }
}

// set_print_options(PrintOptions {
//     threshold: 1000,    // Default or custom threshold for summarization.
//     edge_items: 3,      // Default or custom edge items to display.
//     precision: Some(8), // High value for full precision.
// });
// println!("{}", output);

#[test]
fn test_linear_ones() {
    let seed = 42u64;
    let input_dim = 2;
    let output_dim = 4;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "linear.weight".to_string(),
        ArrayWrapper(normal_array(seed + 1, &[input_dim, output_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "linear.bias".to_string(),
        ArrayWrapper(normal_array(seed, &[output_dim], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = LinearModule::<TestBackend>::new(&device);
    model.load_from(&mut store).unwrap();
    let input_data = constant_array(&vec![input_dim], 1.0).to_tensor_data().unwrap();
    let input_tensor = Tensor::from_data(input_data, &device);

    let output = model.forward(input_tensor);
    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [-1.77364016, 1.29809809, -0.31307063, -1.68842816],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[test]
fn test_linear_uniform() {
    let seed = 44u64;
    let input_dim = 2;
    let output_dim = 4;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "linear.weight".to_string(),
        ArrayWrapper(normal_array(seed + 1, &[input_dim, output_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "linear.bias".to_string(),
        ArrayWrapper(normal_array(seed, &[output_dim], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = LinearModule::<TestBackend>::new(&device);
    model.load_from(&mut store).unwrap();

    let input_data = uniform_array(seed + 2, &[input_dim], 0.0, 1.0).to_tensor_data().unwrap();
    let input_tensor = Tensor::from_data(input_data, &device);
    let output = model.forward(input_tensor);

    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [-0.53813028, -1.69855022, 0.92013592, 0.92915082],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

// --- Adversarial weight-loading tests ---
// These test burn's SafetensorsStore behaviour against malformed weights.
// If a simple Linear module catches these, larger models will too.

#[test]
fn load_wrong_weight_shape_reports_error() {
    let seed = 42u64;
    let output_dim = 4;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    // weight should be [2, 4] but we provide [3, 5]
    tensors
        .insert("linear.weight".to_string(), ArrayWrapper(normal_array(seed, &[3, 5], 0.0, 1.0)));
    tensors.insert(
        "linear.bias".to_string(),
        ArrayWrapper(normal_array(seed + 1, &[output_dim], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let device = Default::default();
    let mut model = LinearModule::<TestBackend>::new(&device);
    let result = model.load_from(&mut store);
    // Burn may return Err or Ok with errors in ApplyResult
    match result {
        Err(_) => {} // shape mismatch caught as hard error
        Ok(apply) => {
            assert!(
                !apply.is_success() || !apply.errors.is_empty(),
                "loading wrong-shape weights should not succeed silently"
            );
        }
    }
}

#[test]
fn load_missing_weight_key_returns_error() {
    let seed = 42u64;
    let output_dim = 4;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    // Only provide bias, omit weight entirely
    tensors.insert(
        "linear.bias".to_string(),
        ArrayWrapper(normal_array(seed, &[output_dim], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let device = Default::default();
    let mut model = LinearModule::<TestBackend>::new(&device);
    let result = model.load_from(&mut store);
    assert!(result.is_err(), "omitting weight tensor should return an error");
}

#[test]
fn load_missing_bias_key_returns_error() {
    let seed = 42u64;
    let input_dim = 2;
    let output_dim = 4;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    // Only provide weight, omit bias
    tensors.insert(
        "linear.weight".to_string(),
        ArrayWrapper(normal_array(seed, &[input_dim, output_dim], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let device = Default::default();
    let mut model = LinearModule::<TestBackend>::new(&device);
    let result = model.load_from(&mut store);
    assert!(result.is_err(), "omitting bias tensor should return an error");
}

#[test]
fn load_extra_unused_key_reports_unused() {
    let seed = 42u64;
    let input_dim = 2;
    let output_dim = 4;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "linear.weight".to_string(),
        ArrayWrapper(normal_array(seed, &[input_dim, output_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "linear.bias".to_string(),
        ArrayWrapper(normal_array(seed + 1, &[output_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "linear.bogus_param".to_string(),
        ArrayWrapper(normal_array(seed + 2, &[3], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let device = Default::default();
    let mut model = LinearModule::<TestBackend>::new(&device);
    let result = model.load_from(&mut store).unwrap();
    assert!(
        !result.unused.is_empty(),
        "extra tensor key should be reported as unused, got: {:?}",
        result
    );
}

#[test]
fn load_empty_safetensors_returns_error() {
    let tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    let st = serialize(tensors, &None).unwrap();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let device = Default::default();
    let mut model = LinearModule::<TestBackend>::new(&device);
    let result = model.load_from(&mut store);
    assert!(result.is_err(), "empty safetensors should return an error");
}

#[test]
fn load_no_bytes_returns_error() {
    let mut store = SafetensorsStore::from_bytes(None);
    let device = Default::default();
    let mut model = LinearModule::<TestBackend>::new(&device);
    let result = model.load_from(&mut store);
    assert!(result.is_err(), "loading from None bytes should return an error");
}

#[test]
fn load_corrupted_bytes_returns_error() {
    let garbage: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03];
    let mut store = SafetensorsStore::from_bytes(Some(garbage));
    let device = Default::default();
    let mut model = LinearModule::<TestBackend>::new(&device);
    let result = model.load_from(&mut store);
    assert!(result.is_err(), "loading corrupted bytes should return an error");
}
