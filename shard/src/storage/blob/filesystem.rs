use async_trait::async_trait;
use bytes::Bytes;
use std::{fs, path::Path};

use crate::error::{ShardError, ShardResult};

use super::BlobStorage;

#[derive(Clone)]
pub struct FilesystemBlobStorage {}

impl FilesystemBlobStorage {
    fn new() -> Self {
        Self {}
    }
}

// TODO: switch this to rely on an actor system instead with oneshots to access the file system

// #[async_trait]
// impl BlobStorage for FilesystemBlobStorage {
//     async fn put_object(&self, path: &Path, contents: Bytes) -> ShardResult<()> {
//         if let Some(parent) = path.parent() {
//             fs::create_dir_all(parent).map_err(|e| ShardError::IOError(e.to_string()))?;
//         }
//         fs::write(path, contents).map_err(|e| ShardError::IOError(e.to_string()))?;
//         Ok(())
//     }
//     async fn get_object(&self, path: &Path) -> ShardResult<Bytes> {
//         let contents = fs::read(path).map_err(|e| ShardError::IOError(e.to_string()))?;
//         Ok(Bytes::from(contents))
//     }
//     async fn delete_object(&self, path: &Path) -> ShardResult<()> {
//         fs::remove_file(path).map_err(|e| ShardError::IOError(e.to_string()))
//     }
// }
