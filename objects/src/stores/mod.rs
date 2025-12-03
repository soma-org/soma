use async_trait::async_trait;
use object_store::ObjectStore;
use std::sync::Arc;
use types::{
    error::ObjectResult,
    metadata::{DownloadMetadata, Metadata, ObjectPath},
};

pub mod filesystem;
pub mod memory;

#[async_trait]
pub trait PersistentStore<S>: Send + Sync + Sized + 'static
where
    S: ObjectStore + 'static,
{
    fn object_store(&self) -> &Arc<S>;
    async fn download_metadata(
        &self,
        path: ObjectPath,
        metadata: Metadata,
    ) -> ObjectResult<DownloadMetadata>;
}

#[async_trait]
pub trait EphemeralStore<S>: Send + Sync + Sized + 'static
where
    S: ObjectStore + 'static,
{
    type Buffer: AsRef<[u8]> + Send + Sync;
    fn object_store(&self) -> &Arc<S>;
    async fn buffer_object(&self, path: ObjectPath) -> ObjectResult<Self::Buffer>;
}
