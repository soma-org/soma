use arrgen::{constant_array, normal_array, uniform_array};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayTensor;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Int, PrintOptions, Tensor, TensorPrimitive, Tolerance, set_print_options};
use probes::tensor::{ArrayWrapper, IntoTensorData};
use probes::v1::probe::ProbeConfig;
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
    let seq_len = 4usize;
    let num_heads = 2usize;
    let head_dim = 2usize;
    let num_layers = 2;
    let embedding_dim = head_dim * num_heads;
    let hidden_dim = embedding_dim * 2;

    let device = Default::default();
    let mut model = ProbeConfig::new()
        .with_embedding_dim(embedding_dim)
        .with_pwff_hidden_dim(hidden_dim)
        .with_num_layers(num_layers)
        .with_num_heads(num_heads)
        .with_dropout_rate(0.0)
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
            ArrayWrapper(normal_array(lseed + 1, &vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.norm_1.beta", l).to_string(),
            ArrayWrapper(normal_array(lseed + 2, &vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.query.weight", l).to_string(),
            ArrayWrapper(normal_array(
                lseed + 3,
                &vec![embedding_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.query.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 4, &vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.key.weight", l).to_string(),
            ArrayWrapper(normal_array(
                lseed + 5,
                &vec![embedding_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.key.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 6, &vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.value.weight", l).to_string(),
            ArrayWrapper(normal_array(
                lseed + 7,
                &vec![embedding_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.value.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 8, &vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.output.weight", l).to_string(),
            ArrayWrapper(normal_array(
                lseed + 9,
                &vec![embedding_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("encoder.layers.{}.attention.output.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 10, &vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.norm_2.gamma", l).to_string(),
            ArrayWrapper(normal_array(lseed + 11, &vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.norm_2.beta", l).to_string(),
            ArrayWrapper(normal_array(lseed + 12, &vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_inner.weight", l).to_string(),
            ArrayWrapper(normal_array(
                lseed + 13,
                &vec![embedding_dim, hidden_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_inner.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 14, &vec![hidden_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_outer.weight", l).to_string(),
            ArrayWrapper(normal_array(
                lseed + 15,
                &vec![hidden_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("encoder.layers.{}.pwff.linear_outer.bias", l).to_string(),
            ArrayWrapper(normal_array(lseed + 16, &vec![embedding_dim], 0.0, 1.0)),
        );
    }

    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    model.load_from(&mut store).unwrap();

    let input_data = normal_array(
        seed + 100,
        &vec![batch_size, seq_len, embedding_dim],
        0.0,
        1.0,
    )
    .to_tensor_data()
    .unwrap();
    let input_tensor: Tensor<TestBackend, 3> = Tensor::from_data(input_data, &device);

    let output = model.forward(input_tensor);
    set_print_options(PrintOptions {
        threshold: 1000,    // Default or custom threshold for summarization.
        edge_items: 3,      // Default or custom edge items to display.
        precision: Some(8), // High value for full precision.
    });
    println!("{}", output);

    let expected_output = Tensor::<TestBackend, 3>::from_floats(
        [[
            [5.63600826, 4.15462685, 10.13569641, -3.19333267],
            [3.72393632, 3.78364635, 9.71327305, -2.26928043],
            [5.81568480, 5.00127506, 10.68091393, -2.86526680],
            [1.72769535, 3.66892815, 9.59317303, -2.37384295],
        ]],
        &device,
    );

    output
        .to_data()
        .assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
