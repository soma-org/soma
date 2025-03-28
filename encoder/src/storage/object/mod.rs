pub(crate) mod filesystem;
pub(crate) mod memory;

use crate::error::ShardResult;
use async_trait::async_trait;
use bytes::Bytes;
use std::str::FromStr;

use shared::checksum::Checksum;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug)]
pub enum ServedObjectResponse {
    Direct(Bytes),
    Redirect(String),
}

#[async_trait]
pub(crate) trait ObjectStorage: Send + Sync + Sized + 'static {
    async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ShardResult<()>;
    async fn get_object(&self, path: &ObjectPath) -> ShardResult<Bytes>;
    async fn delete_object(&self, path: &ObjectPath) -> ShardResult<()>;
    async fn serve_object(&self, path: &ObjectPath) -> ShardResult<ServedObjectResponse>;
}

#[async_trait]
pub(crate) trait ObjectSignedUrl: Send + Sync + 'static {
    async fn get_signed_url(&self, path: &ObjectPath) -> ShardResult<String>;
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
