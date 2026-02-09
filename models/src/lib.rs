use std::sync::Arc;

use async_trait::async_trait;
use burn::{Tensor, data::dataloader::DataLoader, prelude::Backend, tensor::TensorData};
use types::error::ModelResult;

pub mod tensor;
pub mod v1;

pub struct ModelOutput<B: Backend> {
    pub embedding: Tensor<B, 1>,
    pub loss: Tensor<B, 1>,
}

pub trait ModelAPI: Send + 'static {
    type Data: Send + 'static;
    type Backend: Backend;
    fn call(&self, data: Self::Data) -> ModelResult<ModelOutput<Self::Backend>>;
}
