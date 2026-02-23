#![allow(clippy::unwrap_used, clippy::expect_used)]

use arrgen::{constant_array, normal_array, uniform_array};
use burn::backend::NdArray;
use burn::module::Param;
use burn::nn::Initializer;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{PrintOptions, Tensor, TensorPrimitive, Tolerance, set_print_options};
use burn::{module::Module, prelude::Backend};
use models::tensor_conversions::{ArrayWrapper, IntoTensorData};
use safetensors::serialize;
use std::collections::HashMap;

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

#[derive(Module, Debug)]
pub struct ParamModule1D<B: Backend> {
    param: Param<Tensor<B, 1>>,
}

impl<B: Backend> ParamModule1D<B> {
    pub fn new(device: &B::Device) -> Self {
        ParamModule1D { param: Initializer::Ones.init([4], device) }
    }

    pub fn forward(&self) -> Tensor<B, 1> {
        self.param.val()
    }
}

#[derive(Module, Debug)]
pub struct ParamModule3D<B: Backend> {
    param: Param<Tensor<B, 3>>,
}

impl<B: Backend> ParamModule3D<B> {
    pub fn new(device: &B::Device) -> Self {
        ParamModule3D { param: Initializer::Ones.init([1, 1, 4], device) }
    }

    pub fn forward(&self) -> Tensor<B, 3> {
        self.param.val()
    }
}
// set_print_options(PrintOptions {
//     threshold: 1000,    // Default or custom threshold for summarization.
//     edge_items: 3,      // Default or custom edge items to display.
//     precision: Some(8), // High value for full precision.
// });
// println!("{}", output);

#[test]
fn test_1d_param() {
    let seed = 42u64;
    let embedding_dim = 4;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors
        .insert("param".to_string(), ArrayWrapper(normal_array(seed, &[embedding_dim], 0.0, 1.0)));
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = ParamModule1D::<TestBackend>::new(&device);
    model.load_from(&mut store).unwrap();
    let output = model.forward();
    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [0.06942791, 0.13293812, 0.26257637, -0.22530088],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}

#[test]
fn test_3d_param() {
    let seed = 42u64;
    let embedding_dim = 4;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "param".to_string(),
        ArrayWrapper(normal_array(seed, &[1, 1, embedding_dim], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = ParamModule3D::<TestBackend>::new(&device);
    model.load_from(&mut store).unwrap();
    let output = model.forward();
    let expected_output = Tensor::<TestBackend, 3>::from_floats(
        [[[0.06942791, 0.13293812, 0.26257637, -0.22530088]]],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
