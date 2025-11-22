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

use crate::{
    checksum::Checksum,
    committee::Epoch,
    crypto::{NetworkKeyPair, NetworkPublicKey, NetworkSignature},
    error::{ShardError, ShardResult, SharedError, SharedResult},
    multiaddr::Multiaddr,
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        scope::{Scope, ScopedMessage},
    },
    sync::to_host_port_str,
};

type SizeInBytes = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum ObjectPath {
    Inputs(Epoch, Digest<Shard>, Checksum),
    Embeddings(Epoch, Digest<Shard>, Checksum),
    Probes(Epoch, Checksum),
    Uploads(Checksum),
}

impl ObjectPath {
    pub fn path(&self) -> Path {
        match self {
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
            Self::Uploads(checksum) => Path::from(format!("uploads/{}", checksum)),
        }
    }
    pub fn checksum(&self) -> Checksum {
        match self {
            Self::Inputs(_, _, checksum) => checksum.clone(),
            Self::Embeddings(_, _, checksum) => checksum.clone(),
            Self::Probes(_, checksum) => checksum.clone(),
            Self::Uploads(checksum) => checksum.clone(),
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SignedParams {
    pub prefix: String,
    pub peers: Option<Vec<NetworkPublicKey>>,
    pub expires: u64,
    pub signature: NetworkSignature,
}

#[derive(Serialize, Deserialize)]
struct SignedParamInner {
    prefix: String,
    peers: Option<Vec<NetworkPublicKey>>,
    expires: u64,
}

impl SignedParams {
    pub fn new(
        prefix: String,
        peers: Option<Vec<NetworkPublicKey>>,
        timeout: Duration,
        signer: &NetworkKeyPair,
    ) -> SignedParams {
        let peers = peers.map(|mut e| {
            e.sort();
            e
        });
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let expires = current_time.saturating_add(timeout.as_secs());
        let inner = SignedParamInner {
            prefix: prefix.clone(),
            peers: peers.clone(),
            expires,
        };
        let msg = bcs::to_bytes(&ScopedMessage::new(Scope::SignedUrl, inner)).unwrap();
        let signature = signer.sign(&msg);
        Self {
            prefix,
            peers,
            expires,
            signature,
        }
    }

    pub fn verify(&self, verifier: &NetworkPublicKey) -> SharedResult<()> {
        let peers = self.peers.clone().map(|mut e| {
            e.sort();
            e
        });
        let inner = SignedParamInner {
            prefix: self.prefix.clone(),
            peers,
            expires: self.expires,
        };
        let msg = bcs::to_bytes(&ScopedMessage::new(Scope::SignedUrl, inner))
            .map_err(|e| SharedError::FastCrypto(e.to_string()))?;
        verifier
            .verify(&msg, &self.signature)
            .map_err(|e| SharedError::FastCrypto(e.to_string()))
    }
}

#[enum_dispatch]
pub trait DefaultDownloadMetadataAPI {
    fn url(&self) -> &Url;
    fn metadata(&self) -> &Metadata;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[enum_dispatch(MtlsDownloadMetadataAPI)]
pub enum MtlsDownloadMetadata {
    V1(MtlsDownloadMetadataV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq, PartialOrd, Ord)]
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
