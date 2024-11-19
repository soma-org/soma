use bytes::Bytes;
use crate::error::ShardResult;
use super::BlobCompression;

pub(crate) struct ZstdCompressor {}

impl ZstdCompressor {
    fn new() -> Self {
        Self{}
    }
}


impl BlobCompression for ZstdCompressor {

    fn compress(contents: Bytes) -> ShardResult<Bytes> {
        
    }

    fn decompress(contents: Bytes) -> ShardResult<Bytes> {
        
    }

}
