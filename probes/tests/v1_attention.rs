use std::collections::HashMap;

use arrgen::{constant_array, normal_array, uniform_array};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayTensor;
use burn::module::Module;
use burn::prelude::Backend;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Int, PrintOptions, Tensor, TensorPrimitive, Tolerance, set_print_options};
use probes::tensor::{ArrayWrapper, IntoTensorData};
use probes::v1::modules::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use safetensors::serialize;

type TestBackend = NdArray<f32>;
type FT = FloatElem<TestBackend>;

// set_print_options(PrintOptions {
//     threshold: 1000,    // Default or custom threshold for summarization.
//     edge_items: 3,      // Default or custom edge items to display.
//     precision: Some(8), // High value for full precision.
// });
// println!("{}", output);

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
            &vec![head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.query.bias".to_string(),
        ArrayWrapper(normal_array(
            seed + 2,
            &vec![head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.key.weight".to_string(),
        ArrayWrapper(normal_array(
            seed + 3,
            &vec![head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.key.bias".to_string(),
        ArrayWrapper(normal_array(
            seed + 4,
            &vec![head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.value.weight".to_string(),
        ArrayWrapper(normal_array(
            seed + 5,
            &vec![head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.value.bias".to_string(),
        ArrayWrapper(normal_array(
            seed + 6,
            &vec![head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.output.weight".to_string(),
        ArrayWrapper(normal_array(
            seed + 7,
            &vec![head_dim * num_heads, head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    tensors.insert(
        "mha.output.bias".to_string(),
        ArrayWrapper(normal_array(
            seed + 8,
            &vec![head_dim * num_heads],
            0.0,
            1.0,
        )),
    );
    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    let mut model: MhaModule<TestBackend> = MhaModule::new(head_dim, num_heads, &device);

    model.load_from(&mut store).unwrap();

    let input_data = normal_array(
        seed + 9,
        &vec![batch_size, seq_len, num_heads * head_dim],
        0.0,
        1.0,
    )
    .to_tensor_data()
    .unwrap();
    let input_tensor: Tensor<TestBackend, 3> = Tensor::from_data(input_data, &device);

    let mha_input = MhaInput::new(input_tensor);
    let output = model.forward(mha_input);
    set_print_options(PrintOptions {
        threshold: 1000,    // Default or custom threshold for summarization.
        edge_items: 3,      // Default or custom edge items to display.
        precision: Some(8), // High value for full precision.
    });
    println!("{}", output);

    let expected_output = Tensor::<TestBackend, 3>::from_floats(
        [[
            [0.02807204, 2.13409519, 5.45623493, 13.48927689],
            [3.20324492, 0.78216082, 0.03813785, 1.87561870],
            [3.04302406, 0.51064676, -0.37436891, 0.42570090],
            [3.08121729, 0.57106179, -0.28497183, 0.74396586],
        ]],
        &device,
    );

    output
        .to_data()
        .assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
