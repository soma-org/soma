use burn::tensor::{Tensor, TensorData};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;
use std::{env, path::PathBuf};

type Backend = burn_ndarray::NdArray<f32>;

pub(crate) fn to_burn_dtype(dtype: safetensors::Dtype) -> burn::tensor::DType {
    match dtype {
        safetensors::Dtype::BOOL => burn::tensor::DType::Bool,
        safetensors::Dtype::U8 => burn::tensor::DType::U8,
        safetensors::Dtype::I8 => burn::tensor::DType::I8,
        safetensors::Dtype::I16 => burn::tensor::DType::I16,
        safetensors::Dtype::U16 => burn::tensor::DType::U16,
        safetensors::Dtype::F16 => burn::tensor::DType::F16,
        safetensors::Dtype::I32 => burn::tensor::DType::I32,
        safetensors::Dtype::U32 => burn::tensor::DType::U32,
        safetensors::Dtype::F32 => burn::tensor::DType::F32,
        safetensors::Dtype::F64 => burn::tensor::DType::F64,
        safetensors::Dtype::I64 => burn::tensor::DType::I64,
        safetensors::Dtype::U64 => burn::tensor::DType::U64,
        safetensors::Dtype::BF16 => burn::tensor::DType::BF16,
        // TODO: handle this more gracefully using a result
        _ => panic!("unsupported"),
    }
}
