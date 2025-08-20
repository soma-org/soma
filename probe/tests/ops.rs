//! Testing each burn op against expected values
mod shared;
use burn::{
    module::Module,
    nn::Gelu,
    record::{FullPrecisionSettings, Recorder},
    tensor::{Int, Tensor, TensorData},
};
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use shared::{
    embed::TestEmbed, layer_norm::TestLayerNorm, linear::TestLinear,
    multi_head_attention::TestMultiHeadAttention, tensors::to_burn_dtype,
};
use std::fs::File;
use std::{env, path::PathBuf};

type Backend = burn_ndarray::NdArray<f32>;

#[cfg(feature = "safetensors")]
#[test]
fn test_linear() {
    let device = Default::default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let data_dir = manifest_dir.join("tests").join("safetensors");
    let input_path = data_dir.join("ops_inputs.safetensors");
    let input_file = File::open(input_path).unwrap();
    let input_buffer = unsafe { MmapOptions::new().map(&input_file).unwrap() };
    let input_tensors = SafeTensors::deserialize(&input_buffer).unwrap();
    for seed in 0..3 {
        let model_path = data_dir.join(format!("linear_params_{}.safetensors", seed));

        let output_path = data_dir.join(format!("ops_outputs_{}.safetensors", seed));
        let output_file = File::open(output_path).unwrap();
        let output_buffer = unsafe { MmapOptions::new().map(&output_file).unwrap() };
        let output_tensors = SafeTensors::deserialize(&output_buffer).unwrap();

        let load_args = LoadArgs::new(model_path).with_adapter_type(AdapterType::NoAdapter);

        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model = TestLinear::<Backend>::new(&device).load_record(record);
        for name in input_tensors.names() {
            let input = input_tensors.tensor(name).unwrap();
            let input = TensorData::from_bytes(
                input.data().to_vec(),
                input.shape(),
                to_burn_dtype(input.dtype()),
            );
            let input = Tensor::<Backend, 1>::from_data(input, &device);

            let output = model.forward(input);

            let expected_output = output_tensors.tensor(&format!("linear_{}", name)).unwrap();
            let expected_output = TensorData::from_bytes(
                expected_output.data().to_vec(),
                expected_output.shape(),
                to_burn_dtype(expected_output.dtype()),
            );
            let expected_output = Tensor::<Backend, 1>::from_data(expected_output, &device);

            assert!(output.all_close(expected_output, None, None))
        }
    }
}

#[cfg(feature = "safetensors")]
#[test]
fn test_layer_norm() {
    let device = Default::default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let data_dir = manifest_dir.join("tests").join("safetensors");
    let input_path = data_dir.join("ops_inputs.safetensors");
    let input_file = File::open(input_path).unwrap();
    let input_buffer = unsafe { MmapOptions::new().map(&input_file).unwrap() };
    let input_tensors = SafeTensors::deserialize(&input_buffer).unwrap();
    for seed in 0..3 {
        let model_path = data_dir.join(format!("layer_norm_params_{}.safetensors", seed));

        let output_path = data_dir.join(format!("ops_outputs_{}.safetensors", seed));
        let output_file = File::open(output_path).unwrap();
        let output_buffer = unsafe { MmapOptions::new().map(&output_file).unwrap() };
        let output_tensors = SafeTensors::deserialize(&output_buffer).unwrap();

        let load_args = LoadArgs::new(model_path).with_adapter_type(AdapterType::NoAdapter);

        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model = TestLayerNorm::<Backend>::new(&device).load_record(record);
        for name in input_tensors.names() {
            let input = input_tensors.tensor(name).unwrap();
            let input = TensorData::from_bytes(
                input.data().to_vec(),
                input.shape(),
                to_burn_dtype(input.dtype()),
            );
            let input = Tensor::<Backend, 1>::from_data(input, &device);

            let output = model.forward(input);

            let expected_output = output_tensors
                .tensor(&format!("layer_norm_{}", name))
                .unwrap();
            let expected_output = TensorData::from_bytes(
                expected_output.data().to_vec(),
                expected_output.shape(),
                to_burn_dtype(expected_output.dtype()),
            );
            let expected_output = Tensor::<Backend, 1>::from_data(expected_output, &device);

            assert!(output.all_close(expected_output, None, None))
        }
    }
}

#[cfg(feature = "safetensors")]
#[test]
fn test_multi_head_attention() {
    let device = Default::default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let data_dir = manifest_dir.join("tests").join("safetensors");
    let input_path = data_dir.join("ops_inputs.safetensors");
    let input_file = File::open(input_path).unwrap();
    let input_buffer = unsafe { MmapOptions::new().map(&input_file).unwrap() };
    let input_tensors = SafeTensors::deserialize(&input_buffer).unwrap();
    for seed in 0..3 {
        let model_path = data_dir.join(format!("multi_head_attention_params_{}.safetensors", seed));

        let output_path = data_dir.join(format!("ops_outputs_{}.safetensors", seed));
        let output_file = File::open(output_path).unwrap();
        let output_buffer = unsafe { MmapOptions::new().map(&output_file).unwrap() };
        let output_tensors = SafeTensors::deserialize(&output_buffer).unwrap();

        let load_args = LoadArgs::new(model_path).with_adapter_type(AdapterType::NoAdapter);

        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model = TestMultiHeadAttention::<Backend>::new(&device).load_record(record);
        for name in input_tensors.names() {
            let input = input_tensors.tensor(name).unwrap();
            let input = TensorData::from_bytes(
                input.data().to_vec(),
                input.shape(),
                to_burn_dtype(input.dtype()),
            );
            let input = Tensor::<Backend, 1>::from_data(input, &device);

            let mha_input: Tensor<Backend, 3> = input.clone().reshape([1, 1, 10]).repeat(&[1, 10]);
            println!("{:?}", mha_input);
            let output = model.forward(mha_input);

            let expected_output = output_tensors
                .tensor(&format!("multi_head_attention_{}", name))
                .unwrap();
            let expected_output = TensorData::from_bytes(
                expected_output.data().to_vec(),
                expected_output.shape(),
                to_burn_dtype(expected_output.dtype()),
            );
            let expected_output = Tensor::<Backend, 3>::from_data(expected_output, &device);
            println!("{}\n{}\n{}\n\n", name, output, expected_output);

            assert!(output.all_close(expected_output, None, None))
        }
    }
}

#[cfg(feature = "safetensors")]
#[test]
fn test_embed() {
    let device = Default::default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let data_dir = manifest_dir.join("tests").join("safetensors");
    for seed in 0..3 {
        let model_path = data_dir.join(format!("embed_params_{}.safetensors", seed));

        let output_path = data_dir.join(format!("ops_outputs_{}.safetensors", seed));
        let output_file = File::open(output_path).unwrap();
        let output_buffer = unsafe { MmapOptions::new().map(&output_file).unwrap() };
        let output_tensors = SafeTensors::deserialize(&output_buffer).unwrap();

        let expected_output = output_tensors.tensor("embed").unwrap();
        let expected_output = TensorData::from_bytes(
            expected_output.data().to_vec(),
            expected_output.shape(),
            to_burn_dtype(expected_output.dtype()),
        );
        let expected_output =
            Tensor::<Backend, 2>::from_data(expected_output, &device).unsqueeze::<3>();

        let load_args = LoadArgs::new(model_path).with_adapter_type(AdapterType::NoAdapter);

        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model = TestEmbed::<Backend>::new(&device).load_record(record);

        let input = Tensor::<Backend, 2, Int>::zeros([1, 1], &device);

        let output = model.forward(input);
        assert!(output.all_close(expected_output, None, None))
    }
}

#[cfg(feature = "safetensors")]
#[test]
fn test_gelu() {
    let device = Default::default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let data_dir = manifest_dir.join("tests").join("safetensors");
    let input_path = data_dir.join("ops_inputs.safetensors");
    let input_file = File::open(input_path).unwrap();
    let input_buffer = unsafe { MmapOptions::new().map(&input_file).unwrap() };
    let input_tensors = SafeTensors::deserialize(&input_buffer).unwrap();
    let gelu = Gelu::new();
    for seed in 0..3 {
        let output_path = data_dir.join(format!("ops_outputs_{}.safetensors", seed));
        let output_file = File::open(output_path).unwrap();
        let output_buffer = unsafe { MmapOptions::new().map(&output_file).unwrap() };
        let output_tensors = SafeTensors::deserialize(&output_buffer).unwrap();

        for name in input_tensors.names() {
            let input = input_tensors.tensor(name).unwrap();
            let input = TensorData::from_bytes(
                input.data().to_vec(),
                input.shape(),
                to_burn_dtype(input.dtype()),
            );
            let input = Tensor::<Backend, 1>::from_data(input, &device);

            let output = gelu.forward(input);

            let expected_output = output_tensors.tensor(&format!("gelu_{}", name)).unwrap();
            let expected_output = TensorData::from_bytes(
                expected_output.data().to_vec(),
                expected_output.shape(),
                to_burn_dtype(expected_output.dtype()),
            );
            let expected_output = Tensor::<Backend, 1>::from_data(expected_output, &device);

            assert!(output.all_close(expected_output, None, None))
        }
    }
}
