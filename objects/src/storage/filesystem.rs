use async_trait::async_trait;
use bytes::Bytes;
use std::path::{Path, PathBuf};
use tokio::{fs, io::BufReader};

use types::error::{ObjectError, ObjectResult};

use super::{ObjectPath, ObjectStorage};

#[derive(Clone)]
pub struct FilesystemObjectStorage {
    base_path: PathBuf,
}

impl FilesystemObjectStorage {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_owned(),
        }
    }

    fn get_full_path(&self, blob_path: &ObjectPath) -> PathBuf {
        self.base_path.join(&blob_path.path)
    }
}

#[async_trait]
impl ObjectStorage for FilesystemObjectStorage {
    type Reader = BufReader<fs::File>;
    type Writer = fs::File;

    async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ObjectResult<()> {
        let full_path = self.get_full_path(path);

        // Create parent directories if they don't exist
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                ObjectError::ObjectStorage(format!("Failed to create directories: {}", e))
            })?;
        }

        // Write the file contents
        fs::write(&full_path, contents)
            .await
            .map_err(|e| ObjectError::ObjectStorage(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    async fn get_object(&self, path: &ObjectPath) -> ObjectResult<Bytes> {
        let full_path = self.get_full_path(path);

        // Read the file contents
        let contents = fs::read(&full_path).await.map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => ObjectError::NotFound(path.path.clone()),
            _ => ObjectError::ObjectStorage(format!("Failed to read file: {}", e)),
        })?;
        Ok(Bytes::from(contents))
    }

    async fn get_object_writer(&self, path: &ObjectPath) -> ObjectResult<Self::Writer> {
        let full_path = self.get_full_path(path);

        // Ensure the parent directories exist
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                ObjectError::ObjectStorage(format!("Failed to create directories: {}", e))
            })?;
        }

        // Open (or create) the file for writing
        let file = fs::File::create(&full_path)
            .await
            .map_err(|e| ObjectError::ObjectStorage(format!("Failed to create file: {}", e)))?;

        Ok(file)
    }

    async fn delete_object(&self, path: &ObjectPath) -> ObjectResult<()> {
        let full_path = self.get_full_path(path);

        match fs::remove_file(&full_path).await {
            Ok(()) => Ok(()),
            Err(e) => match e.kind() {
                std::io::ErrorKind::NotFound => Ok(()), // Consider file-not-found as success for idempotency
                _ => Err(ObjectError::ObjectStorage(format!(
                    "Failed to delete file: {}",
                    e
                ))),
            },
        }
    }
    async fn stream_object(&self, path: &ObjectPath) -> ObjectResult<Self::Reader> {
        let full_path = self.get_full_path(path);
        let file = fs::File::open(&full_path)
            .await
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => ObjectError::NotFound(path.path.clone()),
                _ => ObjectError::ObjectStorage(format!("Failed to open file: {}", e)),
            })?;
        Ok(BufReader::new(file))
    }

    async fn exists(&self, path: &ObjectPath) -> ObjectResult<()> {
        let full_path = self.get_full_path(path);
        match fs::metadata(&full_path).await {
            Ok(metadata) => {
                if metadata.is_file() {
                    Ok(())
                } else {
                    Err(ObjectError::NotFound(path.path.clone()))
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(ObjectError::NotFound(path.path.clone()))
            }
            Err(e) => Err(ObjectError::ObjectStorage(format!(
                "Failed to check existence: {}",
                e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use tempdir::TempDir;

    #[tokio::test]
    async fn test_filesystem_blob_storage() -> ObjectResult<()> {
        // Create a temporary directory that will be automatically cleaned up
        let temp_dir = TempDir::new("blob_storage_test").expect("Failed to create temp dir");

        // Initialize the storage with temp directory
        let storage = FilesystemObjectStorage::new(temp_dir.path());

        // Create test data
        let test_path = ObjectPath::from_str("test/path/file.txt").unwrap();
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
            Err(ObjectError::NotFound(_)) => Ok(()),
            _ => panic!("Expected NotFound error after deletion"),
        }
    }
}
