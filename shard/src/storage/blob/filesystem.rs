use async_trait::async_trait;
use bytes::Bytes;
use std::path::{Path, PathBuf};
use tokio::fs;

use crate::error::{ShardError, ShardResult};

use super::{BlobPath, BlobStorage};

pub struct FilesystemBlobStorage {
    base_path: PathBuf,
}

impl FilesystemBlobStorage {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_owned(),
        }
    }

    fn get_full_path(&self, blob_path: &BlobPath) -> PathBuf {
        self.base_path.join(&blob_path.path)
    }
}

#[async_trait]
impl BlobStorage for FilesystemBlobStorage {
    async fn put_object(&self, path: &BlobPath, contents: Bytes) -> ShardResult<()> {
        let full_path = self.get_full_path(path);

        // Create parent directories if they don't exist
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                ShardError::BlobStorage(format!("Failed to create directories: {}", e))
            })?;
        }

        // Write the file contents
        fs::write(&full_path, contents)
            .await
            .map_err(|e| ShardError::BlobStorage(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    async fn get_object(&self, path: &BlobPath) -> ShardResult<Bytes> {
        let full_path = self.get_full_path(path);

        // Read the file contents
        let contents = fs::read(&full_path).await.map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => ShardError::NotFound(path.path.clone()),
            _ => ShardError::BlobStorage(format!("Failed to read file: {}", e)),
        })?;

        Ok(Bytes::from(contents))
    }

    async fn delete_object(&self, path: &BlobPath) -> ShardResult<()> {
        let full_path = self.get_full_path(path);

        match fs::remove_file(&full_path).await {
            Ok(()) => Ok(()),
            Err(e) => match e.kind() {
                std::io::ErrorKind::NotFound => Ok(()), // Consider file-not-found as success for idempotency
                _ => Err(ShardError::BlobStorage(format!(
                    "Failed to delete file: {}",
                    e
                ))),
            },
        }
    }
}
