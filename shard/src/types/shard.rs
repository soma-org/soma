use crate::error::ShardError;
use crate::types::manifest::ManifestDigest;
use crate::{
    crypto::{DefaultHashFunction, DIGEST_LENGTH},
    error::ShardResult,
};
use bytes::Bytes;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    hash::{Hash, Hasher},
};

use super::authority_committee::Epoch;
use super::modality::Modality;
use super::network_committee::NetworkIdentityIndex;

/// Contains the manifest digest and leader. By keeping these details
/// secret from the broader network and only sharing with selected shard members
/// we can reduce censorship related attacks that target specific users
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ShardSecret {
    /// the digest that uniquely identifies a manifest
    manifest_digest: ManifestDigest, //TODO: switch to a manifest ref to be more in-line?
    /// the node that will be coordinating communication and selecting the commits
    leader: NetworkIdentityIndex,
}

impl ShardSecret {
    /// creates a new shard secret given a manifest digest and leader
    const fn new(manifest_digest: ManifestDigest, leader: NetworkIdentityIndex) -> Self {
        Self {
            manifest_digest,
            leader,
        }
    }
    /// returns the digest for a shard secret
    fn digest(&self) -> ShardResult<ShardSecretDigest> {
        let serialized: Bytes = bcs::to_bytes(self)
            .map_err(ShardError::SerializationFailure)?
            .into();

        let mut hasher = DefaultHashFunction::new();
        hasher.update(serialized);
        Ok(ShardSecretDigest(hasher.finalize().into()))
    }
}

/// Represents a shard secret
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ShardSecretDigest([u8; DIGEST_LENGTH]);

impl ShardSecretDigest {
    /// lex min
    const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    /// lex max
    const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl Hash for ShardSecretDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<ShardSecretDigest> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: ShardSecretDigest) -> Self {
        Digest::new(hd.0)
    }
}
impl fmt::Display for ShardSecretDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
                .get(0..4)
                .ok_or(fmt::Error)?
        )
    }
}

impl fmt::Debug for ShardSecretDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for ShardSecretDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Uniquely identifies a shard by the epoch, leader, entropy, and modality
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ShardRef {
    /// the epoch that this shard was sampled from, important since committees change each epoch
    epoch: Epoch,
    /// the leader of the shard
    leader: NetworkIdentityIndex,
    /// the tbls threshold signature that acts as a safe source of randomness
    entropy: ShardEntropy,
    /// modality
    modality: Modality,
}

impl ShardRef {
    /// lex min.
    const MIN: Self = Self {
        epoch: 0,
        leader: NetworkIdentityIndex::MIN,
        entropy: ShardEntropy::MIN,
        modality: Modality::text(),
    };

    /// lex max
    const MAX: Self = Self {
        epoch: u64::MAX,
        leader: NetworkIdentityIndex::MAX,
        entropy: ShardEntropy::MAX,
        modality: Modality::video(),
    };

    /// creates a new shard ref
    const fn new(
        epoch: Epoch,
        leader: NetworkIdentityIndex,
        entropy: ShardEntropy,
        modality: Modality,
    ) -> Self {
        Self {
            epoch,
            leader,
            entropy,
            modality,
        }
    }
}

// TODO: re-evaluate formats for production debugging.
impl fmt::Display for ShardRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Shard({},{},{})", self.epoch, self.leader, self.entropy)
    }
}

impl fmt::Debug for ShardRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Shard({},{},{})", self.epoch, self.leader, self.entropy)
    }
}

impl Hash for ShardRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.entropy.0[..8]);
    }
}

/// The source of entropy that is used to sample the shard. Combination of the threshold BLS signature and tx digest
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
struct ShardEntropy([u8; DIGEST_LENGTH]);

impl ShardEntropy {
    /// Lexicographic min digest.
    const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);

    /// Lexicographic max digest.
    const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl Hash for ShardEntropy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<ShardEntropy> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: ShardEntropy) -> Self {
        Digest::new(hd.0)
    }
}
impl fmt::Display for ShardEntropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
                .get(0..4)
                .ok_or(fmt::Error)?
        )
    }
}

impl fmt::Debug for ShardEntropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for ShardEntropy {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
