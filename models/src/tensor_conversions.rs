// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::borrow;

/// A wrapper around raw float data + shape that implements `safetensors::View`.
/// This avoids exposing a specific ndarray version in the public API.
pub struct ArrayWrapper {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl ArrayWrapper {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

impl From<(Vec<f32>, Vec<usize>)> for ArrayWrapper {
    fn from((data, shape): (Vec<f32>, Vec<usize>)) -> Self {
        Self { data, shape }
    }
}

impl safetensors::View for ArrayWrapper {
    fn dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> borrow::Cow<'_, [u8]> {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        };
        borrow::Cow::Borrowed(bytes)
    }

    fn data_len(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}
