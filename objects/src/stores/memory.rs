use async_trait::async_trait;
use bytes::Bytes;
use object_store::{memory::InMemory, ObjectStore};
use std::sync::Arc;
use types::{
    error::{ObjectError, ObjectResult},
    metadata::{DownloadMetadata, Metadata, ObjectPath},
};

use crate::stores::{EphemeralStore, PersistentStore};

#[async_trait]
pub trait DownloadMetadataGenerator: Send + Sync + 'static {
    async fn download_metadata(
        &self,
        path: ObjectPath,
        metadata: Metadata,
    ) -> ObjectResult<DownloadMetadata>;
}

#[derive(Clone)]
pub struct PersistentInMemoryStore {
    object_store: Arc<InMemory>,
    generator: Arc<dyn DownloadMetadataGenerator>,
}

impl PersistentInMemoryStore {
    pub fn new(object_store: Arc<InMemory>, generator: Arc<dyn DownloadMetadataGenerator>) -> Self {
        Self {
            object_store,
            generator,
        }
    }
}

#[async_trait]
impl PersistentStore<InMemory> for PersistentInMemoryStore {
    fn object_store(&self) -> &Arc<InMemory> {
        &self.object_store
    }
    async fn download_metadata(
        &self,
        path: ObjectPath,
        metadata: Metadata,
    ) -> ObjectResult<DownloadMetadata> {
        self.generator.download_metadata(path, metadata).await
    }
}

#[derive(Clone)]
pub struct EphemeralInMemoryStore {
    object_store: Arc<InMemory>,
}

impl EphemeralInMemoryStore {
    pub fn new(object_store: Arc<InMemory>) -> Self {
        Self { object_store }
    }
}

#[async_trait]
impl EphemeralStore<InMemory> for EphemeralInMemoryStore {
    type Buffer = Bytes;
    fn object_store(&self) -> &Arc<InMemory> {
        &self.object_store
    }
    async fn buffer_object(&self, path: ObjectPath) -> ObjectResult<Self::Buffer> {
        self.object_store
            .get(&path.path())
            .await
            .map_err(|e| ObjectError::StorageFailure(e.to_string()))?
            .bytes()
            .await
            .map_err(|e| ObjectError::StorageFailure(e.to_string()))
    }
}
