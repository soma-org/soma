// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::borrow;

use burn::tensor::TensorData;
use ndarray::ArrayD;
use types::error::ModelResult;

pub trait IntoTensorData {
    fn to_tensor_data(self) -> ModelResult<TensorData>;
}

impl IntoTensorData for ArrayD<f32> {
    fn to_tensor_data(self) -> ModelResult<TensorData> {
        let shape = self.shape().to_vec();
        let (raw_vec, _) = self.into_raw_vec_and_offset();
        let data = TensorData::new(raw_vec, shape);
        Ok(data)
    }
}

pub struct ArrayWrapper(pub ArrayD<f32>);

impl safetensors::View for ArrayWrapper {
    fn dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn data(&self) -> borrow::Cow<'_, [u8]> {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr() as *const u8,
                self.0.len() * std::mem::size_of::<f32>(),
            )
        };
        borrow::Cow::Borrowed(bytes)
    }

    fn data_len(&self) -> usize {
        self.0.len() * std::mem::size_of::<f32>()
    }
}
