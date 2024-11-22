mod compression;
mod encryption;
mod filesystem;

use crate::{error::ShardResult, types::checksum::Checksum};
use async_trait::async_trait;
use bytes::Bytes;

pub(crate) struct BlobPath {
    // TODO: make this better
    path: String,
}

impl BlobPath {
    pub(crate) fn new(path: String) -> ShardResult<Self> {
        // TODO: add path validation according to a protocol
        Ok(Self { path })
    }

    pub(crate) fn from_checksum(checksum: Checksum) -> Self {
        Self {
            path: checksum.to_string(),
        }
    }

    pub(crate) fn path(&self) -> String {
        self.path.clone()
    }
}

#[async_trait]
pub(crate) trait BlobStorage: Send + Sync + Sized + 'static {
    async fn put_object(&self, path: &BlobPath, contents: Bytes) -> ShardResult<()>;
    async fn get_object(&self, path: &BlobPath) -> ShardResult<Bytes>;
    async fn delete_object(&self, path: &BlobPath) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait BlobSignedUrl: Send + Sync + 'static {
    async fn get_signed_url(&self, path: &BlobPath) -> ShardResult<String>;
}

pub(crate) trait BlobEncryption<T>: Send + Sync + Sized + 'static {
    fn encrypt(key: T, contents: Bytes) -> Bytes;
    fn decrypt(key: T, contents: Bytes) -> Bytes;
}

pub(crate) trait BlobCompression: Send + Sync + Sized + 'static {
    fn compress(contents: Bytes) -> ShardResult<Bytes>;
    fn decompress(contents: Bytes) -> ShardResult<Bytes>;
}
