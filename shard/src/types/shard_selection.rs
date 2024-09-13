use crate::{
    crypto::{
        keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
        DefaultHashFunction, DIGEST_LENGTH,
    },
    error::{ShardError, ShardResult},
};

use std::ops::Deref;
use std::sync::Arc;

use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};

use super::{
    scope::{Scope, ScopedMessage},
    shard_commit::SignedShardCommit,
};

use std::{
    fmt,
    hash::{Hash, Hasher},
};
#[derive(Clone, Deserialize, Serialize)]
#[enum_dispatch(ShardSelectionAPI)]
pub enum ShardSelection {
    V1(ShardSelectionV1),
}

#[enum_dispatch]
pub trait ShardSelectionAPI {
    fn commits(&self) -> &[SignedShardCommit];
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ShardSelectionV1 {
    commits: Vec<SignedShardCommit>,
}

impl ShardSelectionV1 {
    pub(crate) fn new(commits: Vec<SignedShardCommit>) -> Self {
        Self { commits }
    }
}

impl ShardSelectionAPI for ShardSelectionV1 {
    fn commits(&self) -> &[SignedShardCommit] {
        &self.commits
    }
}

#[derive(Deserialize, Serialize)]
pub struct SignedShardSelection {
    shard_selection: ShardSelection,
    signature: Bytes,
}

#[derive(Serialize, Deserialize)]
struct InnerShardSelectionDigest([u8; DIGEST_LENGTH]);

fn compute_inner_shard_selection_digest(
    shard_selection: &ShardSelection,
) -> ShardResult<InnerShardSelectionDigest> {
    let mut hasher = DefaultHashFunction::new();
    hasher.update(bcs::to_bytes(shard_selection).map_err(ShardError::SerializationFailure)?);
    Ok(InnerShardSelectionDigest(hasher.finalize().into()))
}

fn to_shard_selection_scoped_message(
    digest: InnerShardSelectionDigest,
) -> ScopedMessage<InnerShardSelectionDigest> {
    ScopedMessage::new(Scope::ShardSelection, digest)
}

fn compute_shard_selection_signature(
    shard_selection: &ShardSelection,
    protocol_keypair: &ProtocolKeyPair,
) -> ShardResult<ProtocolKeySignature> {
    let digest = compute_inner_shard_selection_digest(shard_selection)?;
    let message = bcs::to_bytes(&to_shard_selection_scoped_message(digest))
        .map_err(ShardError::SerializationFailure)?;
    Ok(protocol_keypair.sign(&message))
}
fn verify_shard_selection_signature(
    shard_selection: &ShardSelection,
    signature: &[u8],
    protocol_pubkey: &ProtocolPublicKey,
) -> ShardResult<()> {
    let digest = compute_inner_shard_selection_digest(shard_selection)?;
    let message = bcs::to_bytes(&to_shard_selection_scoped_message(digest))
        .map_err(ShardError::SerializationFailure)?;
    let sig =
        ProtocolKeySignature::from_bytes(signature).map_err(ShardError::MalformedSignature)?;
    protocol_pubkey
        .verify(&message, &sig)
        .map_err(ShardError::SignatureVerificationFailure)
}

impl Deref for SignedShardSelection {
    type Target = ShardSelection;

    fn deref(&self) -> &Self::Target {
        &self.shard_selection
    }
}

#[derive(Clone)]
pub struct VerifiedShardSelection {
    block: Arc<SignedShardSelection>,
    // add digest or request
    serialized: Bytes,
}

impl VerifiedShardSelection {
    pub(crate) fn new(signed_shard_selection: SignedShardSelection, serialized: Bytes) -> Self {
        Self {
            block: Arc::new(signed_shard_selection),
            serialized,
        }
    }
}

/// Digest of a `ShardSelection` which covers the `ShardSelection` in Bytes format.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShardSelectionDigest([u8; DIGEST_LENGTH]);

impl ShardSelectionDigest {
    /// Lexicographic min & max digest.
    pub const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    pub const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl Hash for ShardSelectionDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<ShardSelectionDigest> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: ShardSelectionDigest) -> Self {
        Digest::new(hd.0)
    }
}

impl fmt::Display for ShardSelectionDigest {
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

impl fmt::Debug for ShardSelectionDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for ShardSelectionDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
