use std::cmp::Ordering;

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use crate::{
    checksum::Checksum,
    error::{ShardError, ShardResult, SharedError, SharedResult},
    multiaddr::Multiaddr,
    shard_crypto::{digest::Digest, keys::PeerPublicKey},
};

type SizeInBytes = usize;

#[enum_dispatch]
pub trait MetadataAPI {
    fn checksum(&self) -> Checksum;
    fn size(&self) -> SizeInBytes;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct MetadataV1 {
    checksum: Checksum,
    size: SizeInBytes,
}

impl MetadataV1 {
    pub fn new(checksum: Checksum, size: SizeInBytes) -> Self {
        Self { checksum, size }
    }
}

impl MetadataAPI for MetadataV1 {
    fn checksum(&self) -> Checksum {
        self.checksum
    }
    fn size(&self) -> SizeInBytes {
        self.size
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[enum_dispatch(MetadataAPI)]
pub enum Metadata {
    V1(MetadataV1),
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
/////////////////////////////////////////////////////
#[enum_dispatch]
pub trait DownloadableMetadataAPI {
    fn peer(&self) -> PeerPublicKey;
    fn address(&self) -> Multiaddr;
    fn metadata(&self) -> Metadata;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct DownloadableMetadataV1 {
    peer: PeerPublicKey,
    address: Multiaddr,
    metadata: MetadataV1,
}

impl DownloadableMetadataV1 {
    pub fn new(peer: PeerPublicKey, address: Multiaddr, metadata: MetadataV1) -> Self {
        Self {
            peer,
            address,
            metadata,
        }
    }
}

impl DownloadableMetadataAPI for DownloadableMetadataV1 {
    fn peer(&self) -> PeerPublicKey {
        self.peer.clone()
    }
    fn address(&self) -> Multiaddr {
        self.address.clone()
    }
    fn metadata(&self) -> Metadata {
        Metadata::V1(self.metadata.clone())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[enum_dispatch(DownloadableMetadataAPI)]
pub enum DownloadableMetadata {
    V1(DownloadableMetadataV1),
}

/// Tx contains Digest<MetadataCommitment> however there is no way to figure out the inner metadata and nonce values from that.
/// The nonce makes it so that the same identical metadata cannot be detected based on hash.
/// Prior to landing on this solution, a double hash was going to be used except it is deterministic meaning
/// that if a piece of metadata had been submitted earlier, the hash and values would be known to network participants.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct MetadataCommitment {
    downloadable_metadata: DownloadableMetadata,
    nonce: [u8; 32],
}

impl MetadataCommitment {
    pub fn new(downloadable_metadata: DownloadableMetadata, nonce: [u8; 32]) -> Self {
        MetadataCommitment {
            downloadable_metadata,
            nonce,
        }
    }

    pub fn downloadable_metadata(&self) -> DownloadableMetadata {
        self.downloadable_metadata.clone()
    }

    pub fn digest(&self) -> ShardResult<Digest<Self>> {
        Digest::new(self).map_err(ShardError::DigestFailure)
    }
}

pub fn verify_metadata(metadata: &Metadata, max_size: Option<usize>) -> SharedResult<()> {
    if metadata.size() == 0 {
        return Err(SharedError::ValidationError("Size must be non-zero".into()));
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
    Ok(())
}
