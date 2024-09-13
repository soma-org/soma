use crate::{
    crypto::{
        keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
        DefaultHashFunction, DIGEST_LENGTH,
    },
    error::{ShardError, ShardResult},
};

use std::ops::Deref;
use std::sync::Arc;

use crate::types::manifest::Manifest;
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};

use super::{
    scope::{Scope, ScopedMessage},
    shard::ShardRef,
};

use std::{
    fmt,
    hash::{Hash, Hasher},
};

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Deserialize, Serialize)]
#[enum_dispatch(ShardCommitAPI)]
pub(crate) enum ShardCommit {
    V1(ShardCommitV1),
}

/// `ShardCommitAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
trait ShardCommitAPI {
    /// returns the shard ref
    fn shard_ref(&self) -> &ShardRef;
    /// returns the manifest (checksums of the actual embeddings)
    fn manifest(&self) -> &Manifest;
}

#[derive(Clone, Deserialize, Serialize)]
struct ShardCommitV1 {
    /// manifest is the checksum / url references for the embeddings
    /// note the embeddings are not released for download yet so this is
    /// effectively just a commit hash with some additional metadata
    manifest: Manifest,
    /// shard ref, this is important for protecting against replay attacks
    shard_ref: ShardRef,
}

impl ShardCommitV1 {
    /// create a shard commit v1
    pub(crate) const fn new(shard_ref: ShardRef, manifest: Manifest) -> Self {
        Self {
            manifest,
            shard_ref,
        }
    }
}

impl ShardCommitAPI for ShardCommitV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
}

/// Signed version of the shard commit
#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct SignedShardCommit {
    /// contains the
    shard_commit: ShardCommit,
    signature: Bytes,
}

#[derive(Serialize, Deserialize)]
struct InnerShardCommitDigest([u8; DIGEST_LENGTH]);

fn compute_inner_shard_commit_digest(
    shard_commit: &ShardCommit,
) -> ShardResult<InnerShardCommitDigest> {
    let mut hasher = DefaultHashFunction::new();
    hasher.update(bcs::to_bytes(shard_commit).map_err(ShardError::SerializationFailure)?);
    Ok(InnerShardCommitDigest(hasher.finalize().into()))
}

fn to_shard_commit_scoped_message(
    digest: InnerShardCommitDigest,
) -> ScopedMessage<InnerShardCommitDigest> {
    ScopedMessage::new(Scope::ShardCommit, digest)
}

fn compute_shard_commit_signature(
    shard_commit: &ShardCommit,
    protocol_keypair: &ProtocolKeyPair,
) -> ShardResult<ProtocolKeySignature> {
    let digest = compute_inner_shard_commit_digest(shard_commit)?;
    let message = bcs::to_bytes(&to_shard_commit_scoped_message(digest))
        .map_err(ShardError::SerializationFailure)?;
    Ok(protocol_keypair.sign(&message))
}
fn verify_shard_commit_signature(
    shard_commit: &ShardCommit,
    signature: &[u8],
    protocol_pubkey: &ProtocolPublicKey,
) -> ShardResult<()> {
    let digest = compute_inner_shard_commit_digest(shard_commit)?;
    let message = bcs::to_bytes(&to_shard_commit_scoped_message(digest))
        .map_err(ShardError::SerializationFailure)?;
    let sig =
        ProtocolKeySignature::from_bytes(signature).map_err(ShardError::MalformedSignature)?;
    protocol_pubkey
        .verify(&message, &sig)
        .map_err(ShardError::SignatureVerificationFailure)
}

impl Deref for SignedShardCommit {
    type Target = ShardCommit;

    fn deref(&self) -> &Self::Target {
        &self.shard_commit
    }
}

#[derive(Clone)]
struct VerifiedShardCommit {
    block: Arc<SignedShardCommit>,
    // add digest or request
    serialized: Bytes,
}

impl VerifiedShardCommit {
    pub(crate) fn new(signed_shard_commit: SignedShardCommit, serialized: Bytes) -> Self {
        Self {
            block: Arc::new(signed_shard_commit),
            serialized,
        }
    }
}

/// Digest of a `ShardCommit` which covers the `ShardCommit` in Bytes format.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
struct ShardCommitDigest([u8; DIGEST_LENGTH]);

impl ShardCommitDigest {
    /// Lexicographic min & max digest.
    const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl Hash for ShardCommitDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<ShardCommitDigest> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: ShardCommitDigest) -> Self {
        Digest::new(hd.0)
    }
}

impl fmt::Display for ShardCommitDigest {
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

impl fmt::Debug for ShardCommitDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for ShardCommitDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
