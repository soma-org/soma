use std::sync::Arc;

use objects::{
    networking::ObjectNetworkClient,
    storage::{ObjectPath, ObjectStorage},
};
use shared::{
    crypto::keys::PeerPublicKey,
    metadata::{Metadata, MetadataAPI},
};
use soma_network::multiaddr::Multiaddr;

use crate::error::ShardResult;
use async_trait::async_trait;

use crate::actors::{ActorMessage, Processor};

#[derive(Clone)]
pub(crate) struct DownloaderInput {
    peer: PeerPublicKey,
    address: Multiaddr,
    metadata: Metadata,
}

impl DownloaderInput {
    pub(crate) fn new(peer: PeerPublicKey, address: Multiaddr, metadata: Metadata) -> Self {
        Self {
            peer,
            address,
            metadata,
        }
    }
}

pub(crate) struct Downloader<C: ObjectNetworkClient, S: ObjectStorage> {
    client: Arc<C>,
    storage: Arc<S>,
}

impl<C: ObjectNetworkClient, S: ObjectStorage> Downloader<C, S> {
    pub(crate) fn new(concurrency: usize, client: Arc<C>, storage: Arc<S>) -> Self {
        Self { client, storage }
    }
}

#[async_trait]
impl<C: ObjectNetworkClient, S: ObjectStorage> Processor for Downloader<C, S> {
    type Input = DownloaderInput;
    type Output = ();

    async fn process(&self, msg: ActorMessage<Self>) {
        let result: ShardResult<()> = async {
            let input = msg.input;
            let object_path = ObjectPath::from_checksum(input.metadata.checksum());

            // check if the object exists
            if self.storage.exists(&object_path).await?.is_ok() {
                // if it exists, no need to download
                return Ok(());
            }

            // get an object writer for the storage backend
            let writer = self.storage.get_object_writer(&object_path).await?;
            // download the object, streaming it directly into storage
            let _ = self
                .client
                .download_object(writer, &input.peer, &input.address, &input.metadata)
                .await?;

            Ok(())
        }
        .await;
        let _ = msg.sender.send(result);
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
