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
