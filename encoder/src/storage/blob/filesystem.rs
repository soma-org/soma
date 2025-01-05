use async_trait::async_trait;
use bytes::Bytes;
use std::path::{Path, PathBuf};
use tokio::fs;

use crate::error::{ShardError, ShardResult};

use super::{BlobPath, BlobStorage};

#[derive(Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use tempdir::TempDir;

    #[tokio::test]
    async fn test_filesystem_blob_storage() -> ShardResult<()> {
        // Create a temporary directory that will be automatically cleaned up
        let temp_dir = TempDir::new("blob_storage_test").expect("Failed to create temp dir");

        // Initialize the storage with temp directory
        let storage = FilesystemBlobStorage::new(temp_dir.path());

        // Create test data
        let test_path = BlobPath::from_str("test/path/file.txt").unwrap();
        let test_contents = Bytes::from("Hello, World!");

        // Test put_object
        storage
            .put_object(&test_path, test_contents.clone())
            .await?;

        // Test get_object
        let retrieved_contents = storage.get_object(&test_path).await?;
        assert_eq!(retrieved_contents, test_contents);

        // Test delete_object
        storage.delete_object(&test_path).await?;

        // Verify the object is deleted by attempting to get it
        match storage.get_object(&test_path).await {
            Err(ShardError::NotFound(_)) => Ok(()),
            _ => panic!("Expected NotFound error after deletion"),
        }
    }
}
