pub(crate) mod http_network;

use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};

use crate::{
    error::ShardResult,
    storage::object::{ObjectPath, ObjectSignedUrl, ObjectStorage},
    types::{encoder_committee::EncoderIndex, encoder_context::EncoderContext},
};

pub(crate) const GET_OBJECT_TIMEOUT: std::time::Duration = Duration::from_secs(60 * 2);

#[async_trait]
pub(crate) trait ObjectNetworkClient: Send + Sync + Sized + 'static {
    async fn get_object(
        &self,
        peer: EncoderIndex,
        path: &ObjectPath,
        timeout: Duration,
    ) -> ShardResult<Bytes>;
}

#[derive(Debug)]
pub enum GetObjectResponse {
    Direct(Bytes),
    Redirect(String),
}

#[async_trait]
pub(crate) trait ObjectNetworkService: Send + Sync + Sized + 'static {
    async fn handle_get_object(
        &self,
        peer: EncoderIndex,
        path: &ObjectPath,
    ) -> ShardResult<GetObjectResponse>;
}

#[derive(Clone)]
pub struct DirectNetworkService<S: ObjectStorage> {
    storage: Arc<S>,
}

impl<S: ObjectStorage> DirectNetworkService<S> {
    pub(crate) fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
}
pub struct SignedNetworkService<S: ObjectStorage + ObjectSignedUrl> {
    storage: Arc<S>,
}

#[async_trait]
impl<S: ObjectStorage> ObjectNetworkService for DirectNetworkService<S> {
    async fn handle_get_object(
        &self,
        peer: EncoderIndex,
        path: &ObjectPath,
    ) -> ShardResult<GetObjectResponse> {
        let bytes = self.storage.get_object(path).await?;
        Ok(GetObjectResponse::Direct(bytes))
    }
}

#[async_trait]
impl<S: ObjectStorage + ObjectSignedUrl> ObjectNetworkService for SignedNetworkService<S> {
    async fn handle_get_object(
        &self,
        peer: EncoderIndex,
        path: &ObjectPath,
    ) -> ShardResult<GetObjectResponse> {
        let url = self.storage.get_signed_url(path).await?;
        Ok(GetObjectResponse::Redirect(url))
    }
}

pub(crate) trait ObjectNetworkManager<S>: Send + Sync + Sized
where
    S: ObjectNetworkService,
{
    /// type alias
    type Client: ObjectNetworkClient;

    fn new(context: Arc<EncoderContext>) -> ShardResult<Self>;
    /// Returns a client
    fn client(&self) -> Arc<Self::Client>;
    /// Starts the network services
    async fn start(&mut self, service: Arc<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
