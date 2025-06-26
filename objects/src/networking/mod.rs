pub mod downloader;
pub mod http_network;

use async_trait::async_trait;
use soma_tls::AllowPublicKeys;
use std::sync::Arc;
use tokio::io::AsyncWrite;

use crate::{
    parameters::Parameters,
    storage::{ObjectPath, ObjectStorage},
};
use shared::error::ObjectResult;
use shared::{
    crypto::keys::{PeerKeyPair, PeerPublicKey},
    metadata::DownloadableMetadata,
};
use soma_network::multiaddr::Multiaddr;

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
        _peer: &PeerPublicKey,
        path: &ObjectPath,
    ) -> ObjectResult<S::Reader> {
        // perform any additional verification, rate limiting, etc.
        self.storage.stream_object(path).await
    }
}

pub trait ObjectNetworkManager<S>: Send + Sync + Sized
where
    S: ObjectStorage,
{
    /// type alias
    type Client: ObjectNetworkClient;

    fn new(
        own_key: PeerKeyPair,
        parameters: Arc<Parameters>,
        allower: AllowPublicKeys,
    ) -> ObjectResult<Self>;
    /// Returns a client
    fn client(&self) -> Arc<Self::Client>;
    /// Starts the network services
    async fn start(&mut self, address: &Multiaddr, service: ObjectNetworkService<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
