use std::collections::HashMap;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::RwLock;

use crate::error::{ShardError, ShardResult};

use super::{ObjectPath, ObjectStorage};

pub struct MemoryObjectStore {
    store: RwLock<HashMap<ObjectPath, Bytes>>,
}

impl MemoryObjectStore {
    fn new_for_test() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl ObjectStorage for MemoryObjectStore {
    async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ShardResult<()> {
        let _ = self.store.write().insert(path.clone(), contents);
        Ok(())
    }
    async fn get_object(&self, path: &ObjectPath) -> ShardResult<Bytes> {
        let store = self.store.read();
        match store.get(path) {
            Some(bytes) => Ok(bytes.clone()),
            None => Err(ShardError::ObjectStorage("object not found".to_string())),
        }
    }
    async fn delete_object(&self, path: &ObjectPath) -> ShardResult<()> {
        let _ = self.store.write().remove(path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_storage() -> ShardResult<()> {
        let path = ObjectPath::new("test".to_string())?;
        let contents = Bytes::from("test");
        let store = MemoryObjectStore::new_for_test();

        store.put_object(&path, contents.clone()).await?;
        let retrieved = store.get_object(&path).await?;
        assert_eq!(contents, retrieved);
        store.delete_object(&path).await?;

        assert!(store.get_object(&path).await.is_err());

        Ok(())
    }
}
