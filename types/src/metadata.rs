use std::{
    cmp::Ordering,
};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    checksum::Checksum,
    error::{SharedError, SharedResult},
};

type SizeInBytes = usize;

#[enum_dispatch]
pub trait MetadataAPI {
    fn checksum(&self) -> Checksum;
    fn size(&self) -> SizeInBytes;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
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
        self.checksum.clone()
    }
    fn size(&self) -> SizeInBytes {
        self.size
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Hash)]
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

#[enum_dispatch]
pub trait ManifestAPI {
    fn url(&self) -> &Url;
    fn metadata(&self) -> &Metadata;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ManifestV1 {
    url: Url,
    metadata: Metadata,
}

impl ManifestV1 {
    pub fn new(url: Url, metadata: Metadata) -> Self {
        Self { url, metadata }
    }
}

impl ManifestAPI for ManifestV1 {
    fn url(&self) -> &Url {
        &self.url
    }
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Hash)]
#[enum_dispatch(ManifestAPI)]
pub enum Manifest {
    V1(ManifestV1),
}

pub fn verify_metadata(metadata: &Metadata, max_size: Option<SizeInBytes>) -> SharedResult<()> {
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
