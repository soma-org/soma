use arrgen::{constant_array, normal_array, uniform_array};
use burn::tensor::ops::FloatElem;
use burn::tensor::{PrintOptions, Tensor, TensorPrimitive, Tolerance, set_print_options};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
};
use burn_ndarray::NdArrayTensor;
use burn_store::{ModuleSnapshot, SafetensorsStore};
use ndarray_safetensors::TensorViewWithDataBuffer;
use safetensors::serialize;
use std::collections::HashMap;

type TestBackend = burn_ndarray::NdArray<f32>;
type FT = FloatElem<TestBackend>;

#[derive(Module, Debug)]
pub struct LinearModule<B: Backend> {
    pub linear: Linear<B>,
}

impl<B: Backend> LinearModule<B> {
    pub fn new(device: &B::Device) -> Self {
        LinearModule {
            linear: LinearConfig::new(2, 4).init(device),
        }
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
    let mut tensors: HashMap<String, TensorViewWithDataBuffer> = HashMap::new();
    tensors.insert(
        "linear.bias".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(seed, vec![4], 0.0, 1.0)),
    );
    tensors.insert(
        "linear.weight".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(seed + 1, vec![2, 4], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = LinearModule::<TestBackend>::new(&device);
    model.apply_from(&mut store).unwrap();
    let input = constant_array(vec![2], 1.0);
    let nd_tensor = NdArrayTensor::from(input.into_shared());
    let primitive = TensorPrimitive::Float(nd_tensor);
    let input_tensor: Tensor<TestBackend, 1> = Tensor::from_primitive(primitive);
    let output = model.forward(input_tensor);
    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [-1.77364016, 1.29809809, -0.31307063, -1.68842816],
        &device,
    );

    output
        .to_data()
        .assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[test]
fn test_linear_uniform() {
    let seed = 44u64;
    let mut tensors: HashMap<String, TensorViewWithDataBuffer> = HashMap::new();
    tensors.insert(
        "linear.bias".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(seed, vec![4], 0.0, 1.0)),
    );
    tensors.insert(
        "linear.weight".to_string(),
        TensorViewWithDataBuffer::new(&normal_array(seed + 1, vec![2, 4], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = LinearModule::<TestBackend>::new(&device);
    model.apply_from(&mut store).unwrap();

    let input = uniform_array(seed + 2, vec![2], 0.0, 1.0);
    let nd_tensor = NdArrayTensor::from(input.into_shared());
    let primitive = TensorPrimitive::Float(nd_tensor);
    let input_tensor: Tensor<TestBackend, 1> = Tensor::from_primitive(primitive);
    let output = model.forward(input_tensor);

    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [-0.53813028, -1.69855022, 0.92013592, 0.92915082],
        &device,
    );

    output
        .to_data()
        .assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
