use async_trait::async_trait;
use burn::store::SafetensorsStore;
use types::error::BlobResult;

use crate::BlobPath;

pub mod filesystem;
pub mod memory;

#[async_trait]
pub trait BlobLoader: Send + Sync + Sized + 'static {
    type Buffer: AsRef<[u8]> + Send + Sync;
    async fn get_buffer(&self, path: &BlobPath) -> BlobResult<Self::Buffer>;
    async fn safetensor_store(&self, path: &BlobPath) -> BlobResult<SafetensorsStore>;
}
