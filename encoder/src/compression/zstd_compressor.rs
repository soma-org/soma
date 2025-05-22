use bytes::Bytes;
use shared::error::{ShardError, ShardResult};
use zstd::bulk::{compress, decompress};

use super::{Compressor, SizeInBytes};

const MAX_CHUNK_SIZE: usize = 10 * 1024 * 1024;

pub(crate) struct ZstdCompressor {}

impl ZstdCompressor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Compressor for ZstdCompressor {
    fn compress(&self, contents: Bytes) -> ShardResult<Bytes> {
        let compressed = compress(contents.as_ref(), zstd::DEFAULT_COMPRESSION_LEVEL)
            .map_err(|e| ShardError::CompressionFailed(e.to_string()))?;
        Ok(Bytes::from(compressed))
    }

    fn decompress(&self, contents: Bytes, uncompressed_size: SizeInBytes) -> ShardResult<Bytes> {
        let decompressed = decompress(contents.as_ref(), uncompressed_size)
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
            let decompressed = compressor.decompress(compressed, original.len()).unwrap();
            assert_eq!(decompressed, original_bytes);

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

        assert!(compressed.len() < original_bytes.len());
    }
    #[test]
    fn test_compression_error_cases() {
        let original = [0_u8; 2048].to_vec();

        let compressor = ZstdCompressor::new();
        let original_bytes = Bytes::from(original.clone());

        let compressed = compressor.compress(original_bytes.clone()).unwrap();

        assert!(compressor
            .decompress(compressed, original.len() - 1)
            .is_err());
    }
}
