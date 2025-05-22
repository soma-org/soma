#![doc = include_str!("README.md")]

use bytes::Bytes;

use shared::error::ShardResult;

pub(crate) mod zstd_compressor;

type SizeInBytes = usize;

pub(crate) trait Compressor: Send + Sync + Sized + 'static {
    fn compress(&self, contents: Bytes) -> ShardResult<Bytes>;
    fn decompress(&self, contents: Bytes, uncompressed_size: SizeInBytes) -> ShardResult<Bytes>;
}
