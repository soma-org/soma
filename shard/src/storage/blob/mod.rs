mod filesystem;

use crate::error::ShardResult;
use async_trait::async_trait;
use bytes::Bytes;
use std::path::Path;

pub(crate) struct BlobPath {
    path: Path,
}

#[async_trait]
pub(crate) trait BlobStorage: Send + Sync + Sized + 'static {
    async fn put_object(&self, path: &BlobPath, contents: Bytes) -> ShardResult<()>;
    async fn get_object(&self, path: &BlobPath) -> ShardResult<Bytes>;
    async fn delete_object(&self, path: &BlobPath) -> ShardResult<()>;
}
