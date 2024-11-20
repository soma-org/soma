use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::{
    networking::blob::{http_network::BlobHttpClient, BlobNetworkClient},
    types::checksum::Checksum,
};

use super::{ActorMessage, Processor};

pub(crate) struct Downloader {
    semaphore: Semaphore,
    client: BlobHttpClient,
}

impl Processor for Downloader {
    type Input = Checksum;
    type Output = Bytes;

    async fn process(&self, msg: ActorMessage<Self>) {
        let permit = self.semaphore.acquire().await.unwrap();
        tokio::spawn(async move {
            let input = msg.input;
            let object = self.client.get_object(peer, path, timeout).await.unwrap();
            msg.sender.send(object);
            drop(permit);
        });
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
