use std::{path, sync::Arc};

use bytes::Bytes;
use tokio::sync::Semaphore;
use tower::limit::concurrency;

use crate::{
    error::ShardResult,
    networking::blob::{http_network::BlobHttpClient, BlobNetworkClient, GET_OBJECT_TIMEOUT},
    storage::blob::BlobPath,
    types::{checksum::Checksum, network_committee::NetworkingIndex},
};
use async_trait::async_trait;

use super::{ActorMessage, Processor};

pub(crate) struct DownloaderInput {
    peer: NetworkingIndex,
    path: BlobPath,
}

impl DownloaderInput {
    pub(crate) fn new(peer: NetworkingIndex, path: BlobPath) -> Self {
        Self { peer, path }
    }
}

pub(crate) struct Downloader<B: BlobNetworkClient> {
    semaphore: Arc<Semaphore>,
    client: Arc<B>,
}

impl<B: BlobNetworkClient> Downloader<B> {
    pub(crate) fn new(concurrency: usize, client: Arc<B>) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(concurrency)),
            client,
        }
    }
}

#[async_trait]
impl<B: BlobNetworkClient> Processor for Downloader<B> {
    type Input = DownloaderInput;
    type Output = Bytes;

    async fn process(&self, msg: ActorMessage<Self>) {
        let client = self.client.clone();
        if let Ok(permit) = self.semaphore.clone().acquire_owned().await {
            tokio::spawn(async move {
                let input = msg.input;
                let object = client
                    .get_object(input.peer, &input.path, GET_OBJECT_TIMEOUT)
                    .await
                    .unwrap();
                let _ = msg.sender.send(Ok(object));
                drop(permit);
            });
        }
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
