use async_trait::async_trait;
use burn::{Tensor, prelude::Backend, store::SafetensorsStore};
use std::sync::Arc;
use types::error::ModelResult;
pub mod consine_distance;
pub mod select_best;
pub mod tensor_conversions;
pub mod v1;

pub struct ModelOutput<B: Backend> {
    pub embedding: Tensor<B, 1>,
    pub loss: Tensor<B, 1>,
}

#[async_trait]
pub trait ModelAPI: Sync + Send + 'static {
    type Data: Clone + Send + 'static;
    type Backend: Backend;
    async fn build_data(&self, buffer: Arc<[u8]>) -> Self::Data;
    async fn call(
        &self,
        data: Self::Data,
        weights: SafetensorsStore,
    ) -> ModelResult<ModelOutput<Self::Backend>>;
}
