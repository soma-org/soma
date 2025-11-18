use async_trait::async_trait;
use bytes::Bytes;
use std::{ops::Range, time::Duration};
use types::error::ObjectResult;
pub mod store;
pub mod url;

#[async_trait]
pub trait ObjectReader: Send + Sync {
    async fn get_full(&self, timeout: Duration) -> ObjectResult<Bytes>;
    async fn get_range(&self, range: Range<u64>, timeout: Duration) -> ObjectResult<Bytes>;
}
