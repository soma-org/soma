use std::{
    cmp::Ordering,
    fmt,
    str::FromStr,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    checksum::Checksum,
    committee::{AuthorityIndex, Epoch},
    consensus::block::Round,
    crypto::{NetworkKeyPair, NetworkPublicKey, NetworkSignature},
    error::{ShardError, ShardResult, SharedError, SharedResult},
    multiaddr::Multiaddr,
    p2p::to_host_port_str,
    shard::Shard,
    shard_crypto::{
        digest::Digest,
        scope::{Scope, ScopedMessage},
    },
};

type SizeInBytes = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum ObjectPath {
    Inputs(Epoch, Checksum),
    Probes(Epoch, Checksum),
    Embeddings(Epoch, Digest<Shard>, Checksum),
    Blocks(Epoch, Round, AuthorityIndex, Checksum),
    Tmp(Epoch, Checksum),
}

impl ObjectPath {
    pub fn path(&self) -> String {
        match self {
            Self::Embeddings(epoch, shard_digest, checksum) => {
                format!(
                    "epochs/{}/shards/{}/embeddings/{}",
                    epoch, shard_digest, checksum
                )
            }
            Self::Probes(epoch, checksum) => {
                format!("epochs/{}/probes/{}", epoch, checksum)
            }
            Self::Inputs(epoch, checksum) => {
                format!("epochs/{}/inputs/{}", epoch, checksum)
            }
            Self::Blocks(epoch, round, authority_index, checksum) => {
                format!(
                    "epochs/{}/rounds/{}/authorities/{}/blocks/{}",
                    epoch, round, authority_index, checksum
                )
            }
            Self::Tmp(epoch, checksum) => {
                format!("epochs/{}/tmp/{}", epoch, checksum)
            }
        }
    }
    pub fn checksum(&self) -> Checksum {
        match self {
            Self::Embeddings(_, _, checksum) => checksum.clone(),
            Self::Probes(_, checksum) => checksum.clone(),
            Self::Inputs(_, checksum) => checksum.clone(),
            Self::Blocks(_, _, _, checksum) => checksum.clone(),
            Self::Tmp(_, checksum) => checksum.clone(),
        }
    }
}

impl fmt::Display for ObjectPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.path())
    }
}

#[enum_dispatch]
pub trait MetadataAPI {
    fn path(&self) -> ObjectPath;
    fn checksum(&self) -> Checksum;
    fn size(&self) -> SizeInBytes;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct MetadataV1 {
    path: ObjectPath,
    size: SizeInBytes,
}

impl MetadataV1 {
    pub fn new(path: ObjectPath, size: SizeInBytes) -> Self {
        Self { path, size }
    }
}

impl MetadataAPI for MetadataV1 {
    fn path(&self) -> ObjectPath {
        self.path.clone()
    }
    fn checksum(&self) -> Checksum {
        self.path.checksum()
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
pub trait DownloadableMetadataAPI {
    fn peer(&self) -> Option<NetworkPublicKey>;
    fn params(&self) -> Option<SignedParams>;
    fn address(&self) -> Multiaddr;
    fn metadata(&self) -> Metadata;
    fn url(&self) -> SharedResult<Url>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct DownloadableMetadataV1 {
    peer: Option<NetworkPublicKey>,
    params: Option<SignedParams>,
    address: Multiaddr,
    metadata: Metadata,
}

impl DownloadableMetadataV1 {
    pub fn new(
        peer: Option<NetworkPublicKey>,
        params: Option<SignedParams>,
        address: Multiaddr,
        metadata: Metadata,
    ) -> Self {
        Self {
            peer,
            params,
            address,
            metadata,
        }
    }
}

impl DownloadableMetadataAPI for DownloadableMetadataV1 {
    fn peer(&self) -> Option<NetworkPublicKey> {
        self.peer.clone()
    }
    fn params(&self) -> Option<SignedParams> {
        self.params.clone()
    }
    fn address(&self) -> Multiaddr {
        self.address.clone()
    }
    fn metadata(&self) -> Metadata {
        self.metadata.clone()
    }
    fn url(&self) -> SharedResult<Url> {
        let host_port =
            to_host_port_str(&self.address).map_err(|e| SharedError::UrlError(e.to_string()))?;
        let mut address = format!("https://{host_port}/{}", self.metadata.path().path());
        if let Some(params) = self.params.clone() {
            let query = serde_urlencoded::to_string(params)
                .map_err(|e| SharedError::UrlError(e.to_string()))?;
            address = format!("{}?{}", address, query);
        }
        let url = Url::from_str(&address).unwrap();
        Ok(url)
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

    pub fn digest(&self) -> ShardResult<Digest<Self>> {
        Digest::new(self).map_err(ShardError::DigestFailure)
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
