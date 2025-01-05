use super::BlobCompression;
use crate::error::{ShardError, ShardResult};
use bytes::Bytes;
use zstd::bulk::{compress, decompress};

const MAX_CHUNK_SIZE: usize = 10 * 1024 * 1024;

pub(crate) struct ZstdCompressor {}

impl ZstdCompressor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl BlobCompression for ZstdCompressor {
    fn compress(&self, contents: Bytes) -> ShardResult<Bytes> {
        let compressed = compress(contents.as_ref(), zstd::DEFAULT_COMPRESSION_LEVEL)
            .map_err(|e| ShardError::CompressionFailed(e.to_string()))?;
        Ok(Bytes::from(compressed))
    }

    fn decompress(&self, contents: Bytes) -> ShardResult<Bytes> {
        // TODO: switch to explicitly setting the size
        let decompressed = decompress(contents.as_ref(), MAX_CHUNK_SIZE)
            .map_err(|e| ShardError::CompressionFailed(e.to_string()))?;
        Ok(Bytes::from(decompressed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arbtest::{arbitrary, arbtest};

    #[test]
    fn test_compression_roundtrip() {
        arbtest(|u| {
            // Generate random bytes of varying lengths
            let original: Vec<u8> = u.arbitrary().unwrap();

            let compressor = ZstdCompressor::new();
            let original_bytes = Bytes::from(original.clone());

            // Compress the data
            let compressed = compressor.compress(original_bytes.clone()).unwrap();

            // Decompress and verify
            let decompressed = compressor.decompress(compressed).unwrap();
            assert_eq!(
                decompressed, original_bytes,
                "Decompressed data should match original"
            );

            Ok(())
        })
        .run();
    }

    #[test]
    fn test_compression_size() {
        // random data is hard to effectively compress, but zeros should consistently be smaller
        let original = [0_u8; 2048].to_vec();

        let compressor = ZstdCompressor::new();
        let original_bytes = Bytes::from(original.clone());

        let compressed = compressor.compress(original_bytes.clone()).unwrap();

        assert!(
            compressed.len() < original_bytes.len(),
            "Compressed data is not smaller than original data"
        );
    }
}
