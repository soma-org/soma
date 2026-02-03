use async_trait::async_trait;
use bytes::Bytes;
use std::{ops::Range, time::Duration};
use types::error::BlobResult;
pub mod store;
pub mod url;

#[async_trait]
pub trait BlobReader: Send + Sync {
    async fn get_full(&self, timeout: Duration) -> BlobResult<Bytes>;
    async fn get_range(&self, range: Range<usize>, timeout: Duration) -> BlobResult<Bytes>;
}
