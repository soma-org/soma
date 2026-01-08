use std::time::Duration;
use async_trait::async_trait;
use types::error::InferenceResult;
use crate::inference::{InferenceInput, InferenceOutput};
// pub mod json_client;
pub mod mock;

#[async_trait]
pub trait InferenceEngineAPI: Send + Sync + Sized + 'static
{
    async fn call(&self, input: InferenceInput, timeout: Duration) -> InferenceResult<InferenceOutput>;
}

