pub(crate) mod http_network;

use async_trait::async_trait;
use bytes::Bytes;
use std::{sync::Arc, time::Duration};

use crate::{
    error::ShardResult,
    storage::object::{ObjectPath, ObjectStorage, ServedObjectResponse},
    types::encoder_committee::{EncoderIndex, Epoch},
};

use shared::{
    crypto::keys::{PeerKeyPair, PeerPublicKey},
    metadata::Metadata,
    multiaddr::Multiaddr,
};

// TODO: scale this with size of object?
pub(crate) const GET_OBJECT_TIMEOUT: std::time::Duration = Duration::from_secs(60 * 2);

#[async_trait]
pub(crate) trait ObjectNetworkClient: Send + Sync + Sized + 'static {
    async fn get_object(
        &self,
        peer: &PeerPublicKey,
        address: &Multiaddr,
        metadata: &Metadata,
        timeout: Duration,
    ) -> ShardResult<Bytes>;
}

#[derive(Clone)]
pub struct ObjectNetworkService<S: ObjectStorage> {
    storage: Arc<S>,
}

impl<S: ObjectStorage> ObjectNetworkService<S> {
    pub(crate) fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }
    async fn handle_get_object(
        &self,
        peer: &PeerPublicKey,
        path: &ObjectPath,
    ) -> ShardResult<ServedObjectResponse> {
        // handle auth
        self.storage.serve_object(path).await
    }
}

pub(crate) trait ObjectNetworkManager<S>: Send + Sync + Sized
where
    S: ObjectStorage,
{
    /// type alias
    type Client: ObjectNetworkClient;

    fn new(peer_keypair: Arc<PeerKeyPair>) -> ShardResult<Self>;
    /// Returns a client
    fn client(&self) -> Arc<Self::Client>;
    /// Starts the network services
    async fn start(&mut self, address: &Multiaddr, service: ObjectNetworkService<S>);
    /// Stops the network services
    async fn stop(&mut self);
}
