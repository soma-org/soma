use std::{ops::Range, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use object_store::ObjectStore;
use types::{
    error::{ObjectError, ObjectResult},
    metadata::ObjectPath,
};

use crate::readers::ObjectReader;

pub struct ObjectStoreReader {
    store: Arc<dyn ObjectStore>,
    object_path: ObjectPath,
}

impl ObjectStoreReader {
    pub fn new(store: Arc<dyn ObjectStore>, object_path: ObjectPath) -> Self {
        Self { store, object_path }
    }
}

#[async_trait]
impl ObjectReader for ObjectStoreReader {
    async fn get_full(&self, timeout: Duration) -> ObjectResult<Bytes> {
        match tokio::time::timeout(timeout, async {
            let get_result = self
                .store
                .get(&self.object_path.path())
                .await
                .map_err(ObjectError::ObjectStoreError)?;
            get_result
                .bytes()
                .await
                .map_err(ObjectError::ObjectStoreError)
        })
        .await
        {
            Ok(Ok(bytes)) => Ok(bytes),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(ObjectError::Timeout), // elapsed
        }
    }
    async fn get_range(&self, range: Range<usize>, timeout: Duration) -> ObjectResult<Bytes> {
        let path = self.object_path.path().clone();
        tokio::time::timeout(timeout, self.store.get_range(&path, range))
            .await
            .map_err(|_| ObjectError::Timeout)?
            .map_err(ObjectError::ObjectStoreError)
    }
}
