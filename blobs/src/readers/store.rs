use std::{ops::Range, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use object_store::{ObjectStore, path::Path};
use types::error::{BlobError, BlobResult};

use crate::{BlobPath, readers::BlobReader};

pub struct BlobStoreReader {
    pub store: Arc<dyn ObjectStore>,
    pub path: Path,
}

impl BlobStoreReader {
    pub fn new(store: Arc<dyn ObjectStore>, path: &BlobPath) -> Self {
        Self { store, path: path.path() }
    }
}

#[async_trait]
impl BlobReader for BlobStoreReader {
    async fn get_full(&self, timeout: Duration) -> BlobResult<Bytes> {
        match tokio::time::timeout(timeout, async {
            let get_result =
                self.store.get(&self.path).await.map_err(BlobError::ObjectStoreError)?;
            get_result.bytes().await.map_err(BlobError::ObjectStoreError)
        })
        .await
        {
            Ok(Ok(bytes)) => Ok(bytes),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(BlobError::Timeout), // elapsed
        }
    }
    async fn get_range(&self, range: Range<usize>, timeout: Duration) -> BlobResult<Bytes> {
        tokio::time::timeout(timeout, self.store.get_range(&self.path, range))
            .await
            .map_err(|_| BlobError::Timeout)?
            .map_err(BlobError::ObjectStoreError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use types::checksum::Checksum;

    async fn setup(data: &[u8]) -> BlobStoreReader {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let blob_path = BlobPath::Data(0, Checksum::default());
        store.put(&blob_path.path(), data.to_vec().into()).await.unwrap();
        BlobStoreReader::new(store, &blob_path)
    }

    #[tokio::test]
    async fn get_full_returns_all_bytes() {
        let data = b"hello store reader";
        let reader = setup(data).await;
        let result = reader.get_full(Duration::from_secs(5)).await.unwrap();
        assert_eq!(result.as_ref(), data);
    }

    #[tokio::test]
    async fn get_range_returns_slice() {
        let data = b"0123456789abcdef";
        let reader = setup(data).await;
        let result = reader.get_range(4..10, Duration::from_secs(5)).await.unwrap();
        assert_eq!(result.as_ref(), b"456789");
    }

    #[tokio::test]
    async fn get_full_not_found() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let blob_path = BlobPath::Data(0, Checksum::default());
        let reader = BlobStoreReader::new(store, &blob_path);
        let err = reader.get_full(Duration::from_secs(5)).await.unwrap_err();
        assert!(matches!(err, BlobError::ObjectStoreError(_)));
    }
}
