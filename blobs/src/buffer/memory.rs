use async_trait::async_trait;
use bytes::Bytes;
use object_store::{ObjectStore, memory::InMemory};
use types::error::{BlobError, BlobResult};

use crate::{BlobPath, buffer::BufferAPI};

#[async_trait]
impl BufferAPI for InMemory {
    type Buffer = Bytes;
    async fn get_buffer(&self, path: BlobPath) -> BlobResult<Self::Buffer> {
        self.get(&path.path())
            .await
            .map_err(|e| BlobError::StorageFailure(e.to_string()))?
            .bytes()
            .await
            .map_err(|e| BlobError::StorageFailure(e.to_string()))
    }
}
