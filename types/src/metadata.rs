use std::{
    cmp::Ordering,
    fmt,
    str::FromStr,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use enum_dispatch::enum_dispatch;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use url::Url;
use uuid::Uuid;

use crate::{
    checksum::Checksum,
    committee::Epoch,
    crypto::{NetworkKeyPair, NetworkPublicKey, NetworkSignature},
    error::{ShardError, ShardResult, SharedError, SharedResult},
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        scope::{Scope, ScopedMessage},
    },
};

type SizeInBytes = usize;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum ObjectPath {
    Uploads(Checksum),
    Inputs(Epoch, Digest<Shard>, Checksum),
    Embeddings(Epoch, Digest<Shard>, Checksum),
    Probes(Epoch, Checksum),
    Tmp(Epoch, Digest<Shard>, Uuid),
}

impl ObjectPath {
    pub fn path(&self) -> Path {
        match self {
            Self::Uploads(checksum) => Path::from(format!("uploads/{}", checksum)),
            Self::Inputs(epoch, shard_digest, checksum) => Path::from(format!(
                "epochs/{}/shards/{}/inputs/{}",
                epoch, shard_digest, checksum
            )),
            Self::Embeddings(epoch, shard_digest, checksum) => Path::from(format!(
                "epochs/{}/shards/{}/embeddings/{}",
                epoch, shard_digest, checksum
            )),
            Self::Probes(epoch, checksum) => {
                Path::from(format!("epochs/{}/probes/{}", epoch, checksum))
            }
            Self::Tmp(epoch, shard_digest, uuid) => Path::from(format!(
                "epochs/{}/shards/{}/tmp/{}",
                epoch, shard_digest, uuid
            )),
        }
    }

    pub fn new_tmp(epoch: Epoch, shard_digest: Digest<Shard>) -> Self {
        Self::Tmp(epoch, shard_digest, Uuid::new_v4())
    }
    pub fn etag(&self) -> String {
        match self {
            Self::Uploads(checksum) => checksum.to_string(),
            Self::Inputs(_, _, checksum) => checksum.to_string(),
            Self::Embeddings(_, _, checksum) => checksum.to_string(),
            Self::Probes(_, checksum) => checksum.to_string(),
            Self::Tmp(_, _, uuid) => uuid.to_string(),
        }
    }
}

impl fmt::Display for ObjectPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.path().as_ref())
    }
}

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
pub trait DefaultDownloadMetadataAPI {
    fn url(&self) -> &Url;
    fn metadata(&self) -> &Metadata;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DefaultDownloadMetadataV1 {
    url: Url,
    metadata: Metadata,
}

impl DefaultDownloadMetadataV1 {
    pub fn new(url: Url, metadata: Metadata) -> Self {
        Self { url, metadata }
    }
}

impl DefaultDownloadMetadataAPI for DefaultDownloadMetadataV1 {
    fn url(&self) -> &Url {
        &self.url
    }
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[enum_dispatch(DefaultDownloadMetadataAPI)]
pub enum DefaultDownloadMetadata {
    V1(DefaultDownloadMetadataV1),
}

#[enum_dispatch]
pub trait MtlsDownloadMetadataAPI {
    fn peer(&self) -> &NetworkPublicKey;
    fn url(&self) -> &Url;
    fn metadata(&self) -> &Metadata;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MtlsDownloadMetadataV1 {
    peer: NetworkPublicKey,
    url: Url,
    metadata: Metadata,
}

impl MtlsDownloadMetadataV1 {
    pub fn new(peer: NetworkPublicKey, url: Url, metadata: Metadata) -> Self {
        Self {
            peer,
            url,
            metadata,
        }
    }
}

impl MtlsDownloadMetadataAPI for MtlsDownloadMetadataV1 {
    fn peer(&self) -> &NetworkPublicKey {
        &self.peer
    }
    fn url(&self) -> &Url {
        &self.url
    }
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[enum_dispatch(MtlsDownloadMetadataAPI)]
pub enum MtlsDownloadMetadata {
    V1(MtlsDownloadMetadataV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum DownloadMetadata {
    Default(DefaultDownloadMetadata),
    Mtls(MtlsDownloadMetadata),
}

impl DownloadMetadata {
    pub fn url(&self) -> &Url {
        match self {
            Self::Default(dm) => dm.url(),
            Self::Mtls(dm) => dm.url(),
        }
    }
    pub fn metadata(&self) -> &Metadata {
        match self {
            Self::Default(dm) => dm.metadata(),
            Self::Mtls(dm) => dm.metadata(),
        }
    }
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
