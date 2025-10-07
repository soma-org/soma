use arrgen::{constant_array, normal_array, uniform_array};
use burn::tensor::ops::FloatElem;
use burn::tensor::{Int, PrintOptions, Tensor, TensorPrimitive, Tolerance, set_print_options};
use burn_ndarray::NdArrayTensor;
use burn_store::{ModuleSnapshot, SafetensorsStore};
use ndarray_safetensors::TensorViewWithDataBuffer;
use probes::v1::modules::encoder::EncoderConfig;
use safetensors::serialize;
use std::collections::HashMap;

type TestBackend = burn_ndarray::NdArray<f32>;
type FT = FloatElem<TestBackend>;

// set_print_options(PrintOptions {
//     threshold: 1000,    // Default or custom threshold for summarization.
//     edge_items: 3,      // Default or custom edge items to display.
//     precision: Some(8), // High value for full precision.
// });
// println!("{}", output);

#[test]
fn test_v1_encoder_ones() {
    let seed = 42u64;
    let batch_size = 2usize;
    let seq_len = 1usize;
    let num_heads = 1usize;
    let head_dim = 2usize;
    let num_layers = 1;
    let max_wavelength = 10_000.0;
    let scale_factor = 1.0;
    let embedding_dim = head_dim * num_heads;
    let hidden_dim = embedding_dim * 2;

    let device = Default::default();
    let mut model = EncoderConfig::new()
        .with_embedding_dim(embedding_dim)
        .with_pwff_hidden_dim(hidden_dim)
        .with_num_layers(num_layers)
        .with_num_heads(num_heads)
        .with_dropout_rate(0.0)
        .with_scale_factor(scale_factor)
        .with_max_wavelength(max_wavelength)
        .init(&device);

    let mut tensors: HashMap<String, TensorViewWithDataBuffer> = HashMap::new();

    for l in 0..num_layers {
        let lseed = seed + l as u64;
        tensors.insert(
            format!("layers.{}.norm_1.gamma", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 1, vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.norm_1.beta", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 2, vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.query.weight", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(
                lseed + 3,
                vec![embedding_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("layers.{}.attention.query.bias", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 4, vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.key.weight", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(
                lseed + 5,
                vec![embedding_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("layers.{}.attention.key.bias", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 6, vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.value.weight", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(
                lseed + 7,
                vec![embedding_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("layers.{}.attention.value.bias", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 8, vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.attention.output.weight", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(
                lseed + 9,
                vec![embedding_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("layers.{}.attention.output.bias", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 10, vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.norm_2.gamma", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 11, vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.norm_2.beta", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 12, vec![embedding_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.pwff.linear_inner.weight", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(
                lseed + 13,
                vec![embedding_dim, hidden_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("layers.{}.pwff.linear_inner.bias", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 14, vec![hidden_dim], 0.0, 1.0)),
        );
        tensors.insert(
            format!("layers.{}.pwff.linear_outer.weight", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(
                lseed + 15,
                vec![hidden_dim, embedding_dim],
                0.0,
                1.0,
            )),
        );
        tensors.insert(
            format!("layers.{}.pwff.linear_outer.bias", l).to_string(),
            TensorViewWithDataBuffer::new(&normal_array(lseed + 16, vec![embedding_dim], 0.0, 1.0)),
        );
    }

    let st = serialize(tensors, &None).unwrap();
    let device = Default::default();
    let mut store = SafetensorsStore::from_bytes(Some(st));
    model.apply_from(&mut store).unwrap();

    let inputs = constant_array(vec![batch_size, seq_len, embedding_dim], 1.0);
    let nd_tensor = NdArrayTensor::from(inputs.into_shared());
    let primitive = TensorPrimitive::Float(nd_tensor);
    let input_tensor: Tensor<TestBackend, 3> = Tensor::from_primitive(primitive);
    let positions: Tensor<TestBackend, 2, Int> = Tensor::from_data([[1]], &device);

    let output = model.forward(input_tensor, positions);
    set_print_options(PrintOptions {
        threshold: 1000,    // Default or custom threshold for summarization.
        edge_items: 3,      // Default or custom edge items to display.
        precision: Some(8), // High value for full precision.
    });
    println!("{}", output);

    let expected_output = Tensor::<TestBackend, 3>::from_floats(
        [[[0.6717827, 2.158453]], [[0.6717827, 2.158453]]],
        &device,
    );

    output
        .to_data()
        .assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
}
