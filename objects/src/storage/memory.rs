use std::{collections::HashMap, io::Cursor};

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::RwLock;
use tokio::io::BufReader;

use crate::error::{ObjectError, ObjectResult};

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
    type Reader = BufReader<Cursor<Bytes>>;
    async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ObjectResult<()> {
        let _ = self.store.write().insert(path.clone(), contents);
        Ok(())
    }
    async fn get_object(&self, path: &ObjectPath) -> ObjectResult<Bytes> {
        let store = self.store.read();
        match store.get(path) {
            Some(bytes) => Ok(bytes.clone()),
            None => Err(ObjectError::ObjectStorage("object not found".to_string())),
        }
    }
    async fn delete_object(&self, path: &ObjectPath) -> ObjectResult<()> {
        let _ = self.store.write().remove(path);
        Ok(())
    }
    async fn stream_object(&self, path: &ObjectPath) -> ObjectResult<Self::Reader> {
        let bytes = self.get_object(path).await?;
        Ok(BufReader::new(Cursor::new(bytes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_storage() -> ObjectResult<()> {
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
