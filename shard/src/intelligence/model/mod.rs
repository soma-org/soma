pub(crate) mod python;
use crate::error::ShardResult;
use async_trait::async_trait;
use numpy::ndarray::ArrayD;

/// The `Model` trait establishes a standard set of operations that can be taken
/// on any implementation of a model. There are two primary functions, one to initialize
/// the model, and call to trigger processing.

#[async_trait]
pub(crate) trait Model: Send + Sync {
    /// call takes some data references, sends the data to the model, and receives data references
    /// for the model output in return. Model implementations are expected to interface with the storage
    /// backend directly and only supply references in return.
    async fn call(&self, input: &ArrayD<f32>) -> ShardResult<ArrayD<f32>>;
}
