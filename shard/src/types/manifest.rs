use crate::crypto::{DefaultHashFunction, DIGEST_LENGTH};
use crate::error::{ShardError, ShardResult};
use crate::types::checksum::Checksum;
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
    sync::Arc,
};
use url::Url;

/// size of a chunk in bytes
type SizeInBytes = u64;

/// `ManifestAPI` describes the API for interacting with versioned Manifests
#[enum_dispatch]
trait ManifestAPI {
    /// Returns the signed transaction
    fn chunks(&self) -> &[Chunk];
}

/// Versioned manifest type
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ManifestAPI)]
pub enum Manifest {
    V1(ManifestV1),
}

impl Manifest {
    /// new constructs a new transaction certificate
    fn new_v1(chunks: Vec<Chunk>) -> ManifestV1 {
        ManifestV1 { chunks }
    }
}

/// Version 1 of the manifest. Adding versioning here because while not currently sent over the wire,
/// it is reasonable to assume that it may be.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct ManifestV1 {
    chunks: Vec<Chunk>,
}

impl ManifestAPI for ManifestV1 {
    fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }
}

/// The API to interact with versioned chunks
#[enum_dispatch]
trait ChunkAPI {
    /// Returns the signed transaction
    fn checksum(&self) -> Checksum;
    fn url(&self) -> &Url;
    fn size(&self) -> SizeInBytes;
}

/// Version switch for the chunk versions
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ChunkAPI)]
enum Chunk {
    V1(ChunkV1),
}

/// First version of the chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkV1 {
    /// the hash of the chunk data
    checksum: Checksum,
    /// where the data can be accessed
    url: Url,
    /// the size of the chunk in bytes
    size: SizeInBytes,
}

impl ChunkAPI for ChunkV1 {
    fn checksum(&self) -> Checksum {
        self.checksum
    }
    fn url(&self) -> &Url {
        &self.url
    }
    fn size(&self) -> SizeInBytes {
        self.size
    }
}

macros::generate_digest_type!(Manifest);
macros::generate_verified_type!(Manifest);
