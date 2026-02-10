use arrgen::{constant_array, normal_array};
use burn::backend::NdArray;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Tensor, Tolerance};
use models::tensor_conversions::{ArrayWrapper, IntoTensorData};
use models::v1::modules::pwff::PositionWiseFeedForwardConfig;
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
fn test_v1_pwff_ones() {
    let embedding_dim = 4usize;
    let hidden_dim = 2usize;
    let seed = 42u64;
    let mut tensors: HashMap<String, ArrayWrapper> = HashMap::new();
    tensors.insert(
        "linear_inner.weight".to_string(),
        ArrayWrapper(normal_array(seed + 1, &vec![embedding_dim, hidden_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "linear_inner.bias".to_string(),
        ArrayWrapper(normal_array(seed + 2, &vec![hidden_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "linear_outer.weight".to_string(),
        ArrayWrapper(normal_array(seed + 3, &vec![hidden_dim, embedding_dim], 0.0, 1.0)),
    );
    tensors.insert(
        "linear_outer.bias".to_string(),
        ArrayWrapper(normal_array(seed + 4, &vec![embedding_dim], 0.0, 1.0)),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model = PositionWiseFeedForwardConfig::new()
        .with_dropout_rate(0.0)
        .with_embedding_dim(embedding_dim)
        .with_hidden_dim(hidden_dim)
        .init(&device);
    model.load_from(&mut store).unwrap();
    let input_data = constant_array(&vec![embedding_dim], 1.0).to_tensor_data().unwrap();
    let input_tensor: Tensor<TestBackend, 1> = Tensor::from_data(input_data, &device);
    let output = model.forward(input_tensor);

    let expected_output = Tensor::<TestBackend, 1>::from_floats(
        [0.31442693, 0.70205802, -2.13397980, -1.71679294],
        &device,
    );

    output.to_data().assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
