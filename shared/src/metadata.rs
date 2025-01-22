use crate::{
    checksum::Checksum,
    crypto::{AesKey, EncryptionKey},
};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::digest::Digest;

type SizeInBytes = usize;
type SizeInElements = usize;

/// MetadataAPI is built to at least contain the relevant queries on pieces of Metadata
/// like their checksum, if compression was used the algorithm and uncompressed size,
/// if encryption the algorithm and key digest. Lastly the size to download in bytes.
/// Shape is currently contained for transferring embeddings? but perhaps a better approach
/// can be taken
#[enum_dispatch]
pub trait MetadataAPI {
    fn compression(&self) -> Option<Compression>;
    fn encryption(&self) -> Option<Encryption>;
    fn checksum(&self) -> Checksum;
    fn shape(&self) -> &[SizeInElements];
    fn size(&self) -> SizeInBytes;
}

/// Metadata is the top level wrapper of different versions
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(MetadataAPI)]
pub enum Metadata {
    V1(MetadataV1),
}

impl Metadata {
    /// new constructs a new transaction certificate
    pub fn new_v1(
        compression: Option<CompressionV1>,
        encryption: Option<EncryptionV1>,
        checksum: Checksum,
        shape: Vec<SizeInElements>,
        size: SizeInBytes,
    ) -> Metadata {
        Metadata::V1(MetadataV1 {
            compression,
            encryption,
            checksum,
            shape,
            size,
        })
    }
}
/// Version 1 of the Metadata. Adding versioning here because while not currently sent over the wire,
/// it is reasonable to assume that it may be.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct MetadataV1 {
    // notice that we also version our compression and encryption fields
    compression: Option<CompressionV1>,
    encryption: Option<EncryptionV1>,
    checksum: Checksum,
    shape: Vec<SizeInElements>,
    size: SizeInBytes,
}

impl MetadataAPI for MetadataV1 {
    fn compression(&self) -> Option<Compression> {
        self.compression.map(Compression::V1)
    }
    fn encryption(&self) -> Option<Encryption> {
        self.encryption.map(Encryption::V1)
    }
    fn checksum(&self) -> Checksum {
        self.checksum
    }
    fn shape(&self) -> &[SizeInElements] {
        &self.shape
    }
    fn size(&self) -> SizeInBytes {
        self.size
    }
}

/// The compressionAPI offers a way of accessing the uncompressed size
/// if compression was used. This uncompressed size is useful when decompressing
/// because a buffer can be allocated to the perfect size.
#[enum_dispatch]
pub trait CompressionAPI {
    fn uncompressed_size(&self) -> SizeInBytes;
}

/// Compression is the top level type. Notice that the MetadataAPI returns
/// Compression not CompressionV1
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(CompressionAPI)]
pub enum Compression {
    V1(CompressionV1),
}

/// Algo is versioned independently
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub enum CompressionAlgorithmV1 {
    ZSTD,
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct CompressionV1 {
    algorithm: CompressionAlgorithmV1,
    // common field across all algorithm
    uncompressed_size: SizeInBytes,
}

impl CompressionV1 {
    pub fn new(algorithm: CompressionAlgorithmV1, uncompressed_size: SizeInBytes) -> Self {
        Self {
            algorithm,
            uncompressed_size,
        }
    }
}

impl CompressionAPI for CompressionV1 {
    fn uncompressed_size(&self) -> SizeInBytes {
        self.uncompressed_size
    }
}

/// EncryptionAPI allows us to get the digest (hash) of the encryption key
/// that was used. This allows for authenticating that the right encryption key
/// is being used which is especially useful in reducing networking for the reveal
/// stage of the encoder shard
#[enum_dispatch]
pub trait EncryptionAPI {
    /// key digest only needs to return a type that impl EncryptionKey which allows
    /// the underlying Encryption Enum to store different keys
    fn key_digest(&self) -> &Digest<impl EncryptionKey>;
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(EncryptionAPI)]
pub enum Encryption {
    V1(EncryptionV1),
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub enum EncryptionV1 {
    // notice how different keys types can be used for the unique encryption algos
    Aes256Ctr64LE(Digest<AesKey>),
}

impl EncryptionAPI for EncryptionV1 {
    fn key_digest(&self) -> &Digest<impl EncryptionKey> {
        match self {
            // return the digest but match each key accordingly
            EncryptionV1::Aes256Ctr64LE(digest) => digest,
        }
    }
}
