use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::Semaphore;

use crate::{
    compression::Compressor,
    error::ShardResult,
    networking::object::{http_network::ObjectHttpClient, ObjectNetworkClient, GET_OBJECT_TIMEOUT},
    storage::object::ObjectPath,
};
use async_trait::async_trait;

use crate::actors::{ActorMessage, Processor};

type SizeInBytes = usize;

pub(crate) struct CompressionProcessor<C: Compressor> {
    compressor: Arc<C>,
}

impl<C: Compressor> CompressionProcessor<C> {
    pub(crate) fn new(compressor: Arc<C>) -> Self {
        Self { compressor }
    }
}

pub(crate) enum CompressorInput {
    Compress(Bytes),
    Decompress(Bytes, SizeInBytes),
}

#[async_trait]
impl<C: Compressor> Processor for CompressionProcessor<C> {
    type Input = CompressorInput;
    type Output = Bytes;

    async fn process(&self, msg: ActorMessage<Self>) {
        let compressor = self.compressor.clone();
        let _ = tokio::task::spawn_blocking(move || match msg.input {
            CompressorInput::Compress(contents) => {
                let compression_result = compressor.compress(contents);
                let _ = msg.sender.send(compression_result);
            }
            CompressorInput::Decompress(contents, uncompressed_size) => {
                let decompression_result = compressor.decompress(contents, uncompressed_size);
                let _ = msg.sender.send(decompression_result);
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

#[cfg(test)]
mod tests {
    use arbtest::arbtest;
    use tokio_util::sync::CancellationToken;

    use crate::{actors::ActorManager, compression::zstd_compressor::ZstdCompressor};

    use super::*;

    #[tokio::test]
    async fn test_compression_actor() -> ShardResult<()> {
        let compressor = ZstdCompressor::new();
        let processor = CompressionProcessor::new(Arc::new(compressor));
        let manager = ActorManager::new(1, processor);
        let handle = manager.handle();
        let cancellation_token = CancellationToken::new();
        let original = Bytes::from([0_u8; 2048].to_vec());

        let compressed = handle
            .process(
                CompressorInput::Compress(original.clone()),
                cancellation_token.clone(),
            )
            .await?;
        let decompressed = handle
            .process(
                CompressorInput::Decompress(compressed.clone(), original.len()),
                cancellation_token.clone(),
            )
            .await?;

        assert_eq!(
            decompressed, original,
            "Decompressed data should match original"
        );

        assert!(
            compressed.len() < original.len(),
            "Compressed data is not smaller than original data"
        );

        Ok(())
    }
}
