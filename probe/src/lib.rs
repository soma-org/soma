pub(crate) mod module;

use burn::tensor::{backend::Backend, Tensor, TensorData};
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use module::ByteDecoderV1;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait SerializedProbeAPI {
    fn to_probe<B: Backend>(self, device: &B::Device) -> Probe<B>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(SerializedProbeAPI)]
pub enum SerializedProbe {
    V1(SerializedProbeV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SerializedProbeV1 {
    serialized: Bytes,
}

impl SerializedProbeAPI for SerializedProbeV1 {
    fn to_probe<B: Backend>(self, device: &B::Device) -> Probe<B> {
        Probe::V1(ByteDecoderV1::from_bytes(device, self.serialized))
    }
}

#[enum_dispatch]
pub trait ProbeAPI<B: Backend> {
    fn reconstruction(&self, device: &B::Device, embeddings: Array2<f32>) -> Array2<f32>;
    fn to_serialized_probe(self) -> SerializedProbe;
}

#[derive(Clone, Debug)]
#[enum_dispatch(ProbeAPI<B>)]
pub enum Probe<B: Backend> {
    V1(ByteDecoderV1<B>),
}

impl<B: Backend> Probe<B> {
    pub(crate) fn new_v1(probe: ByteDecoderV1<B>) -> Self {
        Self::V1(probe)
    }
}

impl<B: Backend> ProbeAPI<B> for ByteDecoderV1<B> {
    fn reconstruction(&self, device: &B::Device, embeddings: Array2<f32>) -> Array2<f32> {
        let input_tensor: Tensor<B, 2> = array2_to_tensor(embeddings, device);
        let output_tensor = self.forward(input_tensor);
        tensor_to_array2(output_tensor)
    }

    fn to_serialized_probe(self) -> SerializedProbe {
        let serialized = self.to_bytes();
        SerializedProbe::V1(SerializedProbeV1 { serialized })
    }
}

fn tensor_to_array2<B: Backend>(tensor: Tensor<B, 2>) -> Array2<f32> {
    let shape = tensor.shape();
    let tensor_data = tensor.into_data();
    let data = tensor_data
        .to_vec()
        .expect("Failed to convert TensorData to Vec");
    let rows = shape.dims[0];
    let cols = shape.dims[1];

    Array2::from_shape_vec((rows, cols), data)
        .expect("Failed to create Array2 from tensor data - shape mismatch")
}

fn array2_to_tensor<B: Backend>(array: Array2<f32>, device: &B::Device) -> Tensor<B, 2> {
    let shape = array.shape();
    let rows = shape[0];
    let cols = shape[1];

    let (data, _offset) = array.into_raw_vec_and_offset();

    Tensor::from_data(TensorData::new(data, [rows, cols]), device)
}

#[cfg(test)]
mod tests {
    use super::{array2_to_tensor, tensor_to_array2};
    use burn::{backend::NdArray, tensor::Device};
    use ndarray::{array, Array2};

    #[test]
    fn test_tensor_array_conversion() {
        // Define the backend
        type B = NdArray<f32>;

        // Create a test device
        let device = Device::<B>::default();

        // Create a sample 2D array
        let original_array: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Convert array to tensor
        let tensor = array2_to_tensor::<B>(original_array.clone(), &device);

        // Convert tensor back to array
        let converted_array = tensor_to_array2(tensor);

        // Verify dimensions
        assert_eq!(converted_array.shape(), &[2, 3]);

        // Verify values match
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(
                    converted_array[[i, j]],
                    original_array[[i, j]],
                    "Mismatch at position [{}, {}]",
                    i,
                    j
                );
            }
        }

        // Verify exact equality
        assert_eq!(converted_array, original_array);
    }
}
