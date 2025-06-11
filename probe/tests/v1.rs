//! testing v1 probes
#![cfg(feature = "safetensors")]
mod shared;
use burn::{
    module::Module,
    record::{FullPrecisionSettings, Recorder},
    tensor::{Int, Tensor, TensorData},
};
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};
use memmap2::MmapOptions;
use probe::modules::v1::{decoder::DecoderV1, predictor::PredictorV1};
use safetensors::SafeTensors;
use shared::tensors::to_burn_dtype;
use std::fs::File;
use std::{env, path::PathBuf};

type Backend = burn_ndarray::NdArray<f32>;

#[test]
fn test_predictor() {
    let device = Default::default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let data_dir = manifest_dir.join("tests").join("safetensors");
    let input_path = data_dir.join("predictor_inputs.safetensors");
    let input_file = File::open(input_path).unwrap();
    let input_buffer = unsafe { MmapOptions::new().map(&input_file).unwrap() };
    let input_tensors = SafeTensors::deserialize(&input_buffer).unwrap();

    for seed in 0..3 {
        let model_path = data_dir.join(format!("predictor_v1_params_{}.safetensors", seed));

        let load_args = LoadArgs::new(model_path).with_adapter_type(AdapterType::NoAdapter);

        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model = PredictorV1::<Backend>::init(&device).load_record(record);

        let output_path = data_dir.join(format!("predictor_v1_outputs_{}.safetensors", seed));
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
            let output = model.forward(input);
            let expected_output = output_tensors.tensor(name).unwrap();
            let expected_output = TensorData::from_bytes(
                expected_output.data().to_vec(),
                expected_output.shape(),
                to_burn_dtype(expected_output.dtype()),
            );
            let expected_output = Tensor::<Backend, 1>::from_data(expected_output, &device);
            assert!(output.all_close(expected_output, None, Some(1e-6)))
        }
    }
}

#[test]
fn test_decoder() {
    let device = Default::default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let data_dir = manifest_dir.join("tests").join("safetensors");
    let input_path = data_dir.join("decoder_inputs.safetensors");
    let input_file = File::open(input_path).unwrap();
    let input_buffer = unsafe { MmapOptions::new().map(&input_file).unwrap() };
    let input_tensors = SafeTensors::deserialize(&input_buffer).unwrap();

    for seed in 0..3 {
        let model_path = data_dir.join(format!("decoder_v1_params_{}.safetensors", seed));

        let load_args = LoadArgs::new(model_path).with_adapter_type(AdapterType::NoAdapter);

        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .expect("Should decode state successfully");

        let model = DecoderV1::<Backend>::init(&device).load_record(record);
        let output_path = data_dir.join(format!("decoder_v1_outputs_{}.safetensors", seed));
        let output_file = File::open(output_path).unwrap();
        let output_buffer = unsafe { MmapOptions::new().map(&output_file).unwrap() };
        let output_tensors = SafeTensors::deserialize(&output_buffer).unwrap();
        let byte_input = Tensor::<Backend, 2, Int>::ones([1, 256], &device);
        for name in input_tensors.names() {
            let input = input_tensors.tensor(name).unwrap();
            let input = TensorData::from_bytes(
                input.data().to_vec(),
                input.shape(),
                to_burn_dtype(input.dtype()),
            );
            let input = Tensor::<Backend, 3>::from_data(input, &device);
            let output = model.forward(byte_input.clone(), input);

            let expected_output = output_tensors.tensor(name).unwrap();
            let expected_output = TensorData::from_bytes(
                expected_output.data().to_vec(),
                expected_output.shape(),
                to_burn_dtype(expected_output.dtype()),
            );
            let expected_output = Tensor::<Backend, 3>::from_data(expected_output, &device);
            assert!(output.all_close(expected_output, Some(1e-3), Some(1e-4)))
        }
    }
}
