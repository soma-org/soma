use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::{
    networking::blob::{http_network::BlobHttpClient, BlobNetworkClient, GET_OBJECT_TIMEOUT},
    storage::blob::{BlobCompression, BlobPath},
    types::{
        checksum::Checksum,
        manifest::{Batch, BatchAPI},
        network_committee::NetworkingIndex,
    },
};
use async_trait::async_trait;

use super::{ActorMessage, Processor};

pub(crate) struct Compressor<B: BlobCompression> {
    compressor: Arc<B>,
}

pub(crate) enum CompressorInput {
    Compress(Bytes),
    Decompress(Bytes),
}

#[async_trait]
impl<B: BlobCompression> Processor for Compressor<B> {
    type Input = CompressorInput;
    type Output = Bytes;

    async fn process(&self, msg: ActorMessage<Self>) {
        let compressor = self.compressor.clone();
        let _ = tokio::task::spawn_blocking(move || match msg.input {
            CompressorInput::Compress(contents) => {
                let compressed = compressor.compress(contents).unwrap();
                msg.sender.send(compressed);
            }
            CompressorInput::Decompress(contents) => {
                let decompressed = compressor.decompress(contents).unwrap();
                msg.sender.send(decompressed);
            }
        })
        .await;
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
