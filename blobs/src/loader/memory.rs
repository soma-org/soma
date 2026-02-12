use burn::store::SafetensorsStore;
use bytes::Bytes;
use object_store::{ObjectStore, memory::InMemory};
use types::error::{BlobError, BlobResult};

use crate::{BlobPath, loader::BlobLoader};

#[async_trait::async_trait]
impl BlobLoader for InMemory {
    type Buffer = Bytes;

    async fn get_buffer(&self, path: &BlobPath) -> BlobResult<Self::Buffer> {
        let result = self.get(&path.path()).await.map_err(BlobError::ObjectStoreError)?;
        result.bytes().await.map_err(BlobError::ObjectStoreError)
    }
    async fn safetensor_store(&self, path: &BlobPath) -> BlobResult<SafetensorsStore> {
        let buffer = self.get_buffer(path).await?;
        Ok(SafetensorsStore::from_bytes(Some(buffer.to_vec())))
    }
}
