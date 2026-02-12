use blobs::BlobPath;
use burn::{Tensor, prelude::Backend};
use types::error::ModelResult;

pub mod tensor_conversions;
pub mod v1;

pub struct ModelOutput<B: Backend> {
    pub embedding: Tensor<B, 1>,
    pub loss: Tensor<B, 1>,
}

pub trait ModelAPI: Send + 'static {
    type Data: Send + 'static;
    type Backend: Backend;
    fn call(&self, data: Self::Data, weights: &BlobPath)
    -> ModelResult<ModelOutput<Self::Backend>>;
}
