pub(crate) mod compression;
pub(crate) mod encryption;
pub(crate) mod filesystem;

use crate::{error::ShardResult, types::checksum::Checksum};
use async_trait::async_trait;
use bytes::Bytes;
use std::str::FromStr;

#[derive(Clone)]
pub(crate) struct ObjectPath {
    // TODO: make this better
    path: String,
}

impl ObjectPath {
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

impl FromStr for ObjectPath {
    type Err = crate::error::ShardError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ObjectPath::new(s.to_string())
    }
}

#[async_trait]
pub(crate) trait ObjectStorage: Send + Sync + Sized + 'static {
    async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ShardResult<()>;
    async fn get_object(&self, path: &ObjectPath) -> ShardResult<Bytes>;
    async fn delete_object(&self, path: &ObjectPath) -> ShardResult<()>;
}

#[async_trait]
pub(crate) trait ObjectSignedUrl: Send + Sync + 'static {
    async fn get_signed_url(&self, path: &ObjectPath) -> ShardResult<String>;
}

pub(crate) trait ObjectEncryption<T>: Send + Sync + Sized + 'static {
    fn encrypt(&self, key: T, contents: Bytes) -> Bytes;
    fn decrypt(&self, key: T, contents: Bytes) -> Bytes;
}

pub(crate) trait ObjectCompression: Send + Sync + Sized + 'static {
    fn compress(&self, contents: Bytes) -> ShardResult<Bytes>;
    fn decompress(&self, contents: Bytes) -> ShardResult<Bytes>;
}
