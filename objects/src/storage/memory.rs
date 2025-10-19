use super::{ObjectResult, ObjectStorage};
use crate::storage::memory_file::MemoryFile;
use async_trait::async_trait;
use bytes::Bytes;
use std::{collections::HashMap, sync::Arc};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;
use types::error::ObjectError;
use types::metadata::ObjectPath;

#[derive(Clone, Default)]
pub struct MemoryObjectStore {
    // The store now holds MemoryFile instances, which perfectly mimic Tokio Files.
    pub store: Arc<RwLock<HashMap<ObjectPath, MemoryFile>>>,
}

impl MemoryObjectStore {
    /// Creates a new, empty in-memory object store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ObjectStorage for MemoryObjectStore {
    type Writer = MemoryFile;
    type Reader = BufReader<MemoryFile>;

    async fn get_object_writer(&self, path: &ObjectPath) -> ObjectResult<Self::Writer> {
        // Corrected: Await the write lock before using it.
        let mut store_lock = self.store.write().await;
        let file = store_lock
            .entry(path.clone())
            .or_insert_with(MemoryFile::default);
        Ok(file.clone())
    }

    /// Replaces the entire content of an object with the provided `Bytes`.
    async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ObjectResult<()> {
        let mut new_file = MemoryFile::default();
        new_file
            .write_all(&contents)
            .await
            .map_err(|e| ObjectError::Io(e.to_string(), path.to_string()))?;
        self.store.write().await.insert(path.clone(), new_file);
        Ok(())
    }

    /// Retrieves the full content of an object as `Bytes`.
    async fn get_object(&self, path: &ObjectPath) -> ObjectResult<Bytes> {
        let mut file = self.stream_object(path).await?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .await
            .map_err(|e| ObjectError::Io(e.to_string(), path.to_string()))?;
        Ok(Bytes::from(buffer))
    }

    /// Removes an object from the store.
    async fn delete_object(&self, path: &ObjectPath) -> ObjectResult<()> {
        // Corrected: Await the write lock before calling remove.
        self.store.write().await.remove(path);
        Ok(())
    }

    /// Returns a readable and seekable stream for an object.
    async fn stream_object(&self, path: &ObjectPath) -> ObjectResult<Self::Reader> {
        // Corrected: Await the read lock before calling get.
        let store_lock = self.store.read().await;
        let file = store_lock
            .get(path)
            .cloned()
            .ok_or_else(|| ObjectError::NotFound(path.to_string()))?;
        // Reset the file cursor to the beginning for the new reader.
        let mut handle = file.clone();
        handle
            .seek(std::io::SeekFrom::Start(0))
            .await
            .map_err(|e| ObjectError::Io(e.to_string(), path.to_string()))?;
        Ok(BufReader::new(handle))
    }

    /// Checks for the existence of an object.
    async fn exists(&self, path: &ObjectPath) -> ObjectResult<()> {
        // Corrected: Await the read lock before calling contains_key.
        if self.store.read().await.contains_key(path) {
            Ok(())
        } else {
            Err(ObjectError::NotFound(path.to_string()))
        }
    }
}

// ===== Tests =====
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use types::checksum::Checksum;

    /// Tests the fundamental put, get, and delete operations.
    #[tokio::test]
    async fn test_basic_storage_operations() -> ObjectResult<()> {
        let store = MemoryObjectStore::new();
        let path = ObjectPath::Tmp(0, Checksum::default());
        let contents = Bytes::from("hello, memory store!");

        // 1. Put an object and verify its contents
        store.put_object(&path, contents.clone()).await?;
        let retrieved = store.get_object(&path).await?;
        assert_eq!(contents, retrieved);

        // 2. Overwrite the object and verify again
        let new_contents = Bytes::from("new data");
        store.put_object(&path, new_contents.clone()).await?;
        let retrieved_again = store.get_object(&path).await?;
        assert_eq!(new_contents, retrieved_again);

        // 3. Delete the object and confirm it's gone
        store.delete_object(&path).await?;
        assert!(store.get_object(&path).await.is_err());
        assert!(matches!(
            store.exists(&path).await,
            Err(ObjectError::NotFound(_))
        ));

        Ok(())
    }

    /// Tests the streaming writer and reader functionality.
    #[tokio::test]
    async fn test_streaming_write_and_read() -> ObjectResult<()> {
        let store = MemoryObjectStore::new();
        let path = ObjectPath::Tmp(0, Checksum::default());

        // Get a writer and write data in chunks
        let mut writer = store.get_object_writer(&path).await?;
        writer.write_all(b"part 1, ").await.unwrap();
        writer.write_all(b"part 2.").await.unwrap();
        // The writer (MemoryFile) automatically handles flushing on drop or shutdown.

        // Get a reader and read back the content
        let mut reader = store.stream_object(&path).await?;
        let mut buffer = String::new();
        reader.read_to_string(&mut buffer).await.unwrap();

        assert_eq!(buffer, "part 1, part 2.");

        // Test seeking within the reader
        reader
            .seek(std::io::SeekFrom::Start("part 1, ".len() as u64))
            .await
            .unwrap();
        let mut part2_buffer = String::new();
        reader.read_to_string(&mut part2_buffer).await.unwrap();
        assert_eq!(part2_buffer, "part 2.");

        Ok(())
    }
}
