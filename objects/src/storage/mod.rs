pub mod filesystem;
pub mod memory;
pub mod memory_file;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::io::{AsyncRead, AsyncWrite};

use types::{error::ObjectResult, metadata::ObjectPath};

#[async_trait]
pub trait ObjectStorage: Send + Sync + Sized + 'static {
    type Reader: AsyncRead + Send + Sync;
    type Writer: AsyncWrite + Unpin + Send + Sync;
    async fn put_object(&self, path: &ObjectPath, contents: Bytes) -> ObjectResult<()>;
    async fn get_object(&self, path: &ObjectPath) -> ObjectResult<Bytes>;
    async fn get_object_writer(&self, path: &ObjectPath) -> ObjectResult<Self::Writer>;
    async fn delete_object(&self, path: &ObjectPath) -> ObjectResult<()>;
    async fn stream_object(&self, path: &ObjectPath) -> ObjectResult<Self::Reader>;
    async fn exists(&self, path: &ObjectPath) -> ObjectResult<()>;
}
