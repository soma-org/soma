mod filesystem;

use crate::error::ShardResult;
use async_trait::async_trait;
use bytes::Bytes;

pub(crate) struct BlobPath {
    // TODO: make this better
    path: String,
}

impl BlobPath {
    pub(crate) fn new(path: String) -> ShardResult<Self> {
        // TODO: add path validation according to a protocol
        Ok(Self {
            path
        })
    }
}

#[async_trait]
pub(crate) trait BlobStorage: Send + Sync + Sized + 'static {
    async fn put_object(&self, path: &BlobPath, contents: Bytes) -> ShardResult<()>;
    async fn get_object(&self, path: &BlobPath) -> ShardResult<Bytes>;
    async fn delete_object(&self, path: &BlobPath) -> ShardResult<()>;
}
