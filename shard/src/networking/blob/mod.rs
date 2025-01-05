pub(crate) mod http_network;

use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};

use crate::{
    error::ShardResult,
    storage::blob::{BlobPath, BlobSignedUrl, BlobStorage},
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

#[derive(Debug)]
pub enum GetObjectResponse {
    Direct(Bytes),
    Redirect(String),
}

#[async_trait]
pub(crate) trait BlobNetworkService: Send + Sync + Sized + 'static {
    async fn handle_get_object(
        &self,
        peer: NetworkingIndex,
        path: &BlobPath,
    ) -> ShardResult<GetObjectResponse>;
}

#[derive(Clone)]
pub struct DirectNetworkService<S: BlobStorage> {
    storage: Arc<S>,
}

impl<S: BlobStorage> DirectNetworkService<S> {
    pub(crate) fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}
pub struct SignedNetworkService<S: BlobStorage + BlobSignedUrl> {
    storage: Arc<S>,
}

#[async_trait]
impl<S: BlobStorage> BlobNetworkService for DirectNetworkService<S> {
    async fn handle_get_object(
        &self,
        peer: NetworkingIndex,
        path: &BlobPath,
    ) -> ShardResult<GetObjectResponse> {
        let bytes = self.storage.get_object(path).await?;
        Ok(GetObjectResponse::Direct(bytes))
    }
}

#[async_trait]
impl<S: BlobStorage + BlobSignedUrl> BlobNetworkService for SignedNetworkService<S> {
    async fn handle_get_object(
        &self,
        peer: NetworkingIndex,
        path: &BlobPath,
    ) -> ShardResult<GetObjectResponse> {
        let url = self.storage.get_signed_url(path).await?;
        Ok(GetObjectResponse::Redirect(url))
    }
}

pub(crate) trait BlobNetworkManager<S>: Send + Sync + Sized
where
    S: BlobNetworkService,
{
    /// type alias
    type Client: BlobNetworkClient;

    fn new(context: Arc<EncoderContext>) -> ShardResult<Self>;
    /// Returns a client
    fn client(&self) -> Arc<Self::Client>;
    /// Starts the network services
    async fn start(&mut self, service: Arc<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
