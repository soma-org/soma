#![doc = include_str!("README.md")]

use bytes::Bytes;

use crate::error::ShardResult;

pub(crate) mod zstd_compressor;


pub(crate) trait Compressor: Send + Sync + Sized + 'static {
    fn compress(&self, contents: Bytes) -> ShardResult<Bytes>;
    fn decompress(&self, contents: Bytes) -> ShardResult<Bytes>;
}
