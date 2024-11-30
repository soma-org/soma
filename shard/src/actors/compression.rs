use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::{
    error::ShardResult,
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
                let _ = msg.sender.send(Ok(compressed));
            }
            CompressorInput::Decompress(contents) => {
                let decompressed = compressor.decompress(contents).unwrap();
                let _ = msg.sender.send(Ok(decompressed));
            }
        })
        .await;
        // NOTE: await here causes the program to only operate sequentially because the actor run loop
        // also awaits when calling a processors process fn
    }

    fn shutdown(&mut self) {
        // TODO: check whether to do anything for client shutdown
    }
}
