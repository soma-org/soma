use std::{
    collections::HashMap,
    io::Cursor,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::RwLock;
use tokio::io::{AsyncWrite, BufReader};

use crate::error::{ObjectError, ObjectResult};

use super::{ObjectPath, ObjectStorage};

#[derive(Clone)]
pub struct MemoryObjectStore {
    store: Arc<RwLock<HashMap<ObjectPath, Bytes>>>,
}

impl MemoryObjectStore {
    pub fn new_for_test() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

pub struct MemoryObjectWriter {
    store: Arc<RwLock<HashMap<ObjectPath, Bytes>>>,
    path: ObjectPath,
    buffer: Vec<u8>,
    committed: bool,
}

impl MemoryObjectWriter {
    fn new(store: Arc<RwLock<HashMap<ObjectPath, Bytes>>>, path: ObjectPath) -> Self {
        Self {
            store,
            path,
            buffer: Vec::new(),
            committed: false,
        }
    }

    fn commit(&mut self) {
        if !self.committed {
            let data = std::mem::take(&mut self.buffer);
            self.store
                .write()
                .insert(self.path.clone(), Bytes::from(data));
            self.committed = true;
        }
    }
}

impl AsyncWrite for MemoryObjectWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        self.buffer.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        self.commit();
        Poll::Ready(Ok(()))
    }
}

impl Drop for MemoryObjectWriter {
    fn drop(&mut self) {
        self.commit();
    }
}

#[async_trait]
impl ObjectStorage for MemoryObjectStore {
    type Reader = BufReader<Cursor<Bytes>>;
    type Writer = MemoryObjectWriter;

    async fn get_object_writer(&self, path: &ObjectPath) -> ObjectResult<Self::Writer> {
        Ok(MemoryObjectWriter::new(self.store.clone(), path.clone()))
    }

    async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ObjectResult<()> {
        self.store.write().insert(path.clone(), contents);
        Ok(())
    }

    async fn get_object(&self, path: &ObjectPath) -> ObjectResult<Bytes> {
        self.store
            .read()
            .get(path)
            .cloned()
            .ok_or_else(|| ObjectError::ObjectStorage("object not found".to_string()))
    }

    async fn delete_object(&self, path: &ObjectPath) -> ObjectResult<()> {
        self.store.write().remove(path);
        Ok(())
    }

    async fn stream_object(&self, path: &ObjectPath) -> ObjectResult<Self::Reader> {
        let bytes = self.get_object(path).await?;
        Ok(BufReader::new(Cursor::new(bytes)))
    }

    async fn exists(&self, path: &ObjectPath) -> ObjectResult<()> {
        if self.store.read().contains_key(path) {
            Ok(())
        } else {
            Err(ObjectError::NotFound(path.path.clone()))
        }
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
