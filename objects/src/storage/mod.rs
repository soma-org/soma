pub mod filesystem;
pub mod memory;

use async_trait::async_trait;
use bytes::Bytes;
use std::str::FromStr;
use tokio::io::{AsyncBufRead, AsyncWrite};

use shared::checksum::Checksum;

use crate::error::ObjectResult;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectPath {
    // TODO: make this better
    path: String,
}

impl ObjectPath {
    pub fn new(path: String) -> ObjectResult<Self> {
        // TODO: add path validation according to a protocol
        Ok(Self { path })
    }

    pub fn from_checksum(checksum: Checksum) -> Self {
        Self {
            path: checksum.to_string(),
        }
    }

    pub fn path(&self) -> String {
        self.path.clone()
    }
}

impl FromStr for ObjectPath {
    type Err = crate::error::ObjectError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ObjectPath::new(s.to_string())
    }
}

#[async_trait]
pub trait ObjectStorage: Send + Sync + Sized + 'static {
    type Reader: AsyncBufRead + Send + Sync;
    type Writer: AsyncWrite + Unpin + Send;
    async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ObjectResult<()>;
    async fn get_object(&self, path: &ObjectPath) -> ObjectResult<Bytes>;
    async fn get_object_writer(&self, path: &ObjectPath) -> ObjectResult<Self::Writer>;
    async fn delete_object(&self, path: &ObjectPath) -> ObjectResult<()>;
    async fn stream_object(&self, path: &ObjectPath) -> ObjectResult<Self::Reader>;
    async fn exists(&self, path: &ObjectPath) -> ObjectResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_path() {
        let checksum = Checksum::new_from_bytes(&[1u8; 32]);
        let path = checksum.to_string();

        let path_obj = ObjectPath::new(path.clone()).unwrap();
        let checksum_obj = ObjectPath::from_checksum(checksum);
        assert_eq!(path_obj, checksum_obj);
        assert_eq!(path, checksum_obj.path());
    }
}
