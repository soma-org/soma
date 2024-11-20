use super::BlobCompression;
use crate::error::{ShardError, ShardResult};
use bytes::Bytes;
use zstd::bulk::{compress, decompress};

const MAX_CHUNK_SIZE: usize = 10 * 1024 * 1024;

pub(crate) struct ZstdCompressor {}

impl ZstdCompressor {
    fn new() -> Self {
        Self {}
    }
}

impl BlobCompression for ZstdCompressor {
    fn compress(contents: Bytes) -> ShardResult<Bytes> {
        let compressed = compress(contents.as_ref(), zstd::DEFAULT_COMPRESSION_LEVEL)
            .map_err(|e| ShardError::CompressionFailed(e.to_string()))?;
        Ok(Bytes::from(compressed))
    }

    fn decompress(contents: Bytes) -> ShardResult<Bytes> {
        let decompressed = decompress(contents.as_ref(), MAX_CHUNK_SIZE)
            .map_err(|e| ShardError::CompressionFailed(e.to_string()))?;
        Ok(Bytes::from(decompressed))
    }
}
