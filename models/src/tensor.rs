use burn::{
    Tensor,
    prelude::Backend,
    tensor::{DType, TensorData},
};
use ndarray::ArrayD;
use std::{borrow, ops::Deref};
use types::error::{ModelError, ModelResult};

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

impl IntoTensorData for safetensors::tensor::TensorView<'_> {
    fn to_tensor_data(self) -> ModelResult<TensorData> {
        let bytes = burn::tensor::Bytes::from_bytes_vec(self.data().to_vec());
        Ok(TensorData {
            bytes,
            shape: self.shape().to_vec(),
            dtype: safetensor_dtype_to_burn(self.dtype()).unwrap(),
        })
    }
}

fn dtype_to_safetensors(dtype: DType) -> ModelResult<safetensors::Dtype> {
    use safetensors::Dtype;

    match dtype {
        DType::F64 => Ok(Dtype::F64),
        DType::F32 | DType::Flex32 => Ok(Dtype::F32), // Flex32 is stored as F32
        DType::F16 => Ok(Dtype::F16),
        DType::BF16 => Ok(Dtype::BF16),
        DType::I64 => Ok(Dtype::I64),
        DType::I32 => Ok(Dtype::I32),
        DType::I16 => Ok(Dtype::I16),
        DType::I8 => Ok(Dtype::I8),
        DType::U64 => Ok(Dtype::U64),
        DType::U32 => Ok(Dtype::U32),
        DType::U16 => Err(ModelError::FailedTypeVerification(
            "U16 dtype not yet supported in safetensors".to_string(),
        )),
        DType::U8 => Ok(Dtype::U8),
        DType::Bool => Ok(Dtype::BOOL),
        DType::QFloat(_) => Err(ModelError::FailedTypeVerification(
            "Quantized tensors not yet supported in safetensors".to_string(),
        )),
    }
}

fn safetensor_dtype_to_burn(dtype: safetensors::Dtype) -> ModelResult<DType> {
    use safetensors::Dtype;

    match dtype {
        Dtype::F64 => Ok(DType::F64),
        Dtype::F32 => Ok(DType::F32),
        Dtype::F16 => Ok(DType::F16),
        Dtype::BF16 => Ok(DType::BF16),
        Dtype::I64 => Ok(DType::I64),
        Dtype::I32 => Ok(DType::I32),
        Dtype::I16 => Ok(DType::I16),
        Dtype::I8 => Ok(DType::I8),
        Dtype::U64 => Ok(DType::U64),
        Dtype::U32 => Ok(DType::U32),
        Dtype::U8 => Ok(DType::U8),
        Dtype::BOOL => Ok(DType::Bool),
        _ => Err(ModelError::FailedTypeVerification(format!("Unsupported dtype: {:?}", dtype))),
    }
}

struct TensorWrapper<B: Backend, const D: usize> {
    dtype: DType,
    shape: Vec<usize>,
    tensor: Tensor<B, D>,
}

impl<B: Backend, const D: usize> TensorWrapper<B, D> {
    fn new(tensor: Tensor<B, D>) -> Self {
        Self { dtype: tensor.dtype(), shape: tensor.shape().dims, tensor }
    }
}

impl<B: Backend, const D: usize> TensorWrapper<B, D> {
    fn data_len(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.size()
    }
}

impl<B: Backend, const D: usize> safetensors::View for TensorWrapper<B, D> {
    fn dtype(&self) -> safetensors::Dtype {
        // Convert from burn dtype to safetensors dtype
        dtype_to_safetensors(self.dtype).unwrap_or(safetensors::Dtype::F32)
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> borrow::Cow<'_, [u8]> {
        // Only materialize data when actually needed for serialization
        let data = self.tensor.to_data();
        borrow::Cow::Owned(data.bytes.deref().to_vec())
    }

    fn data_len(&self) -> usize {
        // Use the efficient data_len method from TensorSnapshot
        self.data_len()
    }
}

pub struct ArrayWrapper(pub ArrayD<f32>);

impl safetensors::View for ArrayWrapper {
    fn dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.0.shape()
    }

    fn data(&self) -> borrow::Cow<'_, [u8]> {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr() as *const u8,
                self.0.len() * std::mem::size_of::<f32>(),
            )
        };
        borrow::Cow::Borrowed(bytes)
        // borrow::Cow::Owned(bytes.deref().to_vec())
    }

    fn data_len(&self) -> usize {
        self.0.len() * std::mem::size_of::<f32>()
    }
}
