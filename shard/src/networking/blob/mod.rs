pub(crate) mod http_network;

use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};

use crate::{
    error::ShardResult,
    storage::blob::{BlobPath, BlobStorage},
    types::{context::EncoderContext, network_committee::NetworkingIndex},
};

pub(crate) const GET_OBJECT_TIMEOUT: std::time::Duration = Duration::from_secs(60 * 2);

#[async_trait]
pub(crate) trait BlobNetworkClient: Send + Sync + Sized + 'static {
    async fn get_object(
        &self,
        peer: NetworkingIndex,
        path: &BlobPath,
        timeout: Duration,
    ) -> ShardResult<Bytes>;
}

#[async_trait]
pub(crate) trait BlobNetworkService: Send + Sync + Sized + 'static {
    async fn handle_get_object(&self, peer: NetworkingIndex, path: &BlobPath)
        -> ShardResult<Bytes>;
}

pub(crate) struct BlobStorageNetworkService<S: BlobStorage> {
    blob_storage: Arc<S>,
}

impl<S: BlobStorage> BlobStorageNetworkService<S> {
    fn new(blob_storage: Arc<S>) -> Self {
        Self { blob_storage }
    }
}

#[async_trait]
impl<S: BlobStorage> BlobNetworkService for BlobStorageNetworkService<S> {
    async fn handle_get_object(
        &self,
        peer: NetworkingIndex,
        path: &BlobPath,
    ) -> ShardResult<Bytes> {
        // TODO: handle verification?
        self.blob_storage.get_object(path).await
    }
}

pub(crate) trait BlobNetworkManager<S>: Send + Sync
where
    S: BlobNetworkService,
{
    /// type alias
    type Client: BlobNetworkClient;

    fn new(context: Arc<EncoderContext>) -> Self;
    /// Returns a client
    fn client(&self) -> Arc<Self::Client>;
    /// Starts the network services
    async fn start(&mut self, service: Arc<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
