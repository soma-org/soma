use std::cmp::Ordering;

use crate::{
    checksum::Checksum,
    crypto::EncryptionKey,
    error::{SharedError, SharedResult},
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
    fn size(&self) -> SizeInBytes;
}

/// Metadata is the top level wrapper of different versions
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[enum_dispatch(MetadataAPI)]
pub enum Metadata {
    V1(MetadataV1),
}

impl Metadata {
    pub fn new_for_test(bytes: &[u8]) -> Self {
        Metadata::V1(MetadataV1 {
            compression: None,
            encryption: None,
            checksum: Checksum::new_from_bytes(bytes),
            size: 0,
        })
    }
}

impl Metadata {
    /// new constructs a new transaction certificate
    pub fn new_v1(
        compression: Option<CompressionV1>,
        encryption: Option<EncryptionV1>,
        checksum: Checksum,
        size: SizeInBytes,
    ) -> Metadata {
        Metadata::V1(MetadataV1 {
            compression,
            encryption,
            checksum,
            size,
        })
    }
}
impl PartialOrd for Metadata {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other)) // Delegates to Ord since Checksum is fully comparable
    }
}

// Add Ord implementation
impl Ord for Metadata {
    fn cmp(&self, other: &Self) -> Ordering {
        self.checksum().cmp(&other.checksum())
    }
}
/// Version 1 of the Metadata. Adding versioning here because while not currently sent over the wire,
/// it is reasonable to assume that it may be.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct MetadataV1 {
    // notice that we also version our compression and encryption fields
    compression: Option<CompressionV1>,
    encryption: Option<EncryptionV1>,
    checksum: Checksum,
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
#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum CompressionAlgorithmV1 {
    ZSTD,
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
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
    fn key_digest(&self) -> Digest<EncryptionKey>;
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(EncryptionAPI)]
pub enum Encryption {
    V1(EncryptionV1),
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum EncryptionV1 {
    // notice how different keys types can be used for the unique encryption algos
    Aes256Ctr64LE(Digest<EncryptionKey>),
}

impl EncryptionAPI for EncryptionV1 {
    fn key_digest(&self) -> Digest<EncryptionKey> {
        match self {
            // return the digest but match each key accordingly
            EncryptionV1::Aes256Ctr64LE(digest) => *digest,
        }
    }
}

/// Tx contains Digest<MetadataCommitment> however there is no way to figure out the inner metadata and nonce values from that.
/// The nonce makes it so that the same identical metadata cannot be detected based on hash.
/// Prior to landing on this solution, a double hash was going to be used except it is deterministic meaning
/// that if a piece of metadata had been submitted earlier, the hash and values would be known to network participants.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MetadataCommitment {
    metadata: Metadata,
    nonce: [u8; 32],
}

impl MetadataCommitment {
    pub fn new(metadata: Metadata, nonce: [u8; 32]) -> Self {
        MetadataCommitment { metadata, nonce }
    }

    pub fn metadata(&self) -> Metadata {
        self.metadata.clone()
    }
}

pub fn verify_metadata(
    expected_size: Option<usize>,
    require_compression: Option<bool>,
    require_encryption: Option<bool>,
    max_size: Option<usize>,
) -> impl Fn(&Metadata) -> SharedResult<()> {
    move |metadata: &Metadata| {
        // Basic validations that always run
        if metadata.size() == 0 {
            return Err(SharedError::ValidationError("Size must be non-zero".into()));
        }

        // Check size if specified
        if let Some(expected_size) = expected_size {
            if metadata.size() != expected_size {
                return Err(SharedError::ValidationError(format!(
                    "Size mismatch. Expected {}, got {}",
                    expected_size,
                    metadata.size()
                )));
            }
        }

        // Check max size if specified
        if let Some(max_size) = max_size {
            if metadata.size() > max_size {
                return Err(SharedError::ValidationError(format!(
                    "Size {} exceeds maximum allowed size {}",
                    metadata.size(),
                    max_size
                )));
            }
        }

        // Check compression requirements
        match (metadata.compression(), require_compression) {
            (Some(compression), Some(false)) => {
                return Err(SharedError::ValidationError(
                    "Compression not allowed for this metadata".into(),
                ));
            }
            (None, Some(true)) => {
                return Err(SharedError::ValidationError(
                    "Compression required but not present".into(),
                ));
            }
            (Some(compression), _) => {
                let uncompressed_size = compression.uncompressed_size();
                if uncompressed_size == 0 {
                    return Err(SharedError::ValidationError(
                        "Uncompressed size must be non-zero".into(),
                    ));
                }
                if metadata.size() > uncompressed_size {
                    return Err(SharedError::ValidationError(
                        "Compressed size cannot be larger than uncompressed size".into(),
                    ));
                }
            }
            _ => {}
        }

        // Check encryption requirements
        match (metadata.encryption(), require_encryption) {
            (Some(_), Some(false)) => {
                return Err(SharedError::ValidationError(
                    "Encryption not allowed for this metadata".into(),
                ));
            }
            (None, Some(true)) => {
                return Err(SharedError::ValidationError(
                    "Encryption required but not present".into(),
                ));
            }
            _ => {}
        }

        Ok(())
    }
}
