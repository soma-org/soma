// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use async_trait::async_trait;
use burn::Tensor;
use burn::prelude::Backend;
use burn::store::SafetensorsStore;
use types::error::ModelResult;
pub mod cosine_distance;
pub mod running_mean;
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
    async fn dataloader(&self, buffer: Arc<[u8]>) -> Self::Data;
    async fn eval(
        &self,
        data: Self::Data,
        weights: SafetensorsStore,
        seed: u64,
    ) -> ModelResult<ModelOutput<Self::Backend>>;
}
