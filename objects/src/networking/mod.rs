pub mod downloader;
pub mod http_network;
pub mod proxy;

use async_trait::async_trait;
use soma_tls::{AllowPublicKeys, Allower};
use std::sync::Arc;
use tokio::io::AsyncWrite;
use types::multiaddr::Multiaddr;

use crate::storage::{ObjectPath, ObjectStorage};
use types::error::ObjectResult;
use types::{
    metadata::DownloadableMetadata,
    parameters::Http2Parameters,
    shard_crypto::keys::{PeerKeyPair, PeerPublicKey},
};
// use soma_network::multiaddr::Multiaddr;

#[async_trait]
pub trait ObjectNetworkClient: Send + Sync + 'static {
    async fn download_object<W>(
        &self,
        writer: &mut W,
        downloadable_metadata: &DownloadableMetadata,
    ) -> ObjectResult<()>
    where
        W: AsyncWrite + Unpin + Send;
}

#[derive(Clone)]
pub struct ObjectNetworkService<S: ObjectStorage> {
    storage: Arc<S>,
}

impl<S: ObjectStorage> ObjectNetworkService<S> {
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
    pub(crate) async fn handle_download_object(
        &self,
        path: &ObjectPath,
    ) -> ObjectResult<S::Reader> {
        // perform any additional verification, rate limiting, etc.
        self.storage.stream_object(path).await
    }
}

pub trait ObjectNetworkManager<S, A>: Send + Sync + Sized
where
    S: ObjectStorage,
    A: Allower,
{
    /// type alias
    type Client: ObjectNetworkClient;

    fn new(
        own_key: PeerKeyPair,
        parameters: Arc<Http2Parameters>,
        allower: A,
    ) -> ObjectResult<Self>;
    /// Returns a client
    fn client(&self) -> Arc<Self::Client>;
    /// Starts the network services
    async fn start(&mut self, address: &Multiaddr, service: ObjectNetworkService<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
