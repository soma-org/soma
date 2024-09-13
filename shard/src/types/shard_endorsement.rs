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

use crate::types::{
    scope::{Scope, ScopedMessage},
    score::Score,
    shard::ShardRef,
};

use std::{
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Deserialize, Serialize)]
#[enum_dispatch(ShardEndorsementAPI)]
pub enum ShardEndorsement {
    V1(ShardEndorsementV1),
}

#[enum_dispatch]
pub trait ShardEndorsementAPI {
    fn scores(&self) -> &[Score];
    fn manifest(&self) -> &Manifest;
    fn shard_ref(&self) -> &ShardRef;
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ShardEndorsementV1 {
    scores: Vec<Score>,
    manifest: Manifest,
    shard_ref: ShardRef,
}

impl ShardEndorsementV1 {
    pub(crate) fn new(scores: Vec<Score>, manifest: Manifest, shard_ref: ShardRef) -> Self {
        Self {
            scores,
            manifest,
            shard_ref,
        }
    }
}

impl ShardEndorsementAPI for ShardEndorsementV1 {
    fn scores(&self) -> &[Score] {
        &self.scores
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
}

#[derive(Deserialize, Serialize)]
pub struct SignedShardEndorsement {
    shard_endorsement: ShardEndorsement,
    signature: Bytes,
}

#[derive(Serialize, Deserialize)]
struct InnerShardEndorsementDigest([u8; DIGEST_LENGTH]);

fn compute_inner_shard_endorsement_digest(
    shard_endorsement: &ShardEndorsement,
) -> ShardResult<InnerShardEndorsementDigest> {
    let mut hasher = DefaultHashFunction::new();
    hasher.update(bcs::to_bytes(shard_endorsement).map_err(ShardError::SerializationFailure)?);
    Ok(InnerShardEndorsementDigest(hasher.finalize().into()))
}

fn to_shard_endorsement_scoped_message(
    digest: InnerShardEndorsementDigest,
) -> ScopedMessage<InnerShardEndorsementDigest> {
    ScopedMessage::new(Scope::ShardEndorsement, digest)
}

fn compute_shard_endorsement_signature(
    shard_endorsement: &ShardEndorsement,
    protocol_keypair: &ProtocolKeyPair,
) -> ShardResult<ProtocolKeySignature> {
    let digest = compute_inner_shard_endorsement_digest(shard_endorsement)?;
    let message = bcs::to_bytes(&to_shard_endorsement_scoped_message(digest))
        .map_err(ShardError::SerializationFailure)?;
    Ok(protocol_keypair.sign(&message))
}
fn verify_shard_endorsement_signature(
    shard_endorsement: &ShardEndorsement,
    signature: &[u8],
    protocol_pubkey: &ProtocolPublicKey,
) -> ShardResult<()> {
    let digest = compute_inner_shard_endorsement_digest(shard_endorsement)?;
    let message = bcs::to_bytes(&to_shard_endorsement_scoped_message(digest))
        .map_err(ShardError::SerializationFailure)?;
    let sig =
        ProtocolKeySignature::from_bytes(signature).map_err(ShardError::MalformedSignature)?;
    protocol_pubkey
        .verify(&message, &sig)
        .map_err(ShardError::SignatureVerificationFailure)
}

impl Deref for SignedShardEndorsement {
    type Target = ShardEndorsement;

    fn deref(&self) -> &Self::Target {
        &self.shard_endorsement
    }
}

#[derive(Clone)]
pub struct VerifiedShardEndorsement {
    block: Arc<SignedShardEndorsement>,
    // add digest or request
    serialized: Bytes,
}

impl VerifiedShardEndorsement {
    pub(crate) fn new(signed_shard_endorsement: SignedShardEndorsement, serialized: Bytes) -> Self {
        Self {
            block: Arc::new(signed_shard_endorsement),
            serialized,
        }
    }
}

/// Digest of a `ShardEndorsement` which covers the `ShardEndorsement` in Bytes format.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShardEndorsementDigest([u8; DIGEST_LENGTH]);

impl ShardEndorsementDigest {
    /// Lexicographic min & max digest.
    pub const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    pub const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl Hash for ShardEndorsementDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<ShardEndorsementDigest> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: ShardEndorsementDigest) -> Self {
        Digest::new(hd.0)
    }
}

impl fmt::Display for ShardEndorsementDigest {
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

impl fmt::Debug for ShardEndorsementDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for ShardEndorsementDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
