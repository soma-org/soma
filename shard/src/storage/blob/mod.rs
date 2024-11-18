mod filesystem;

use crate::error::ShardResult;
use bytes::Bytes;
use std::path::Path;

pub(crate) trait BlobStorage: Send + Sync + Clone + 'static {
    // TODO: convert general path to a specific path structure
    fn put_object(&self, path: &Path, data: Bytes) -> ShardResult<()>;
    fn get_object(&self, path: &Path) -> ShardResult<Bytes>;
    fn delete_object(&self, path: &Path) -> ShardResult<()>;
}
