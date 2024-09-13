use crate::{
    crypto::{
        keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
        DefaultHashFunction, DIGEST_LENGTH,
    },
    error::{ShardError, ShardResult},
};

use std::ops::Deref;
use std::sync::Arc;

use crate::types::{
    manifest::Manifest, modality::Modality, shard::ShardSecret,
    transaction_certificate::TransactionCertificate,
};
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};

use super::scope::{Scope, ScopedMessage};
use std::{
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Deserialize, Serialize)]
#[enum_dispatch(ShardInputAPI)]
pub enum ShardInput {
    V1(ShardInputV1),
}

#[enum_dispatch]
pub trait ShardInputAPI {
    fn transaction_certificate(&self) -> &TransactionCertificate;
    fn shard_secret(&self) -> &ShardSecret;
    fn manifest(&self) -> &Manifest;
    fn modality(&self) -> &Modality;
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ShardInputV1 {
    transaction_certificate: TransactionCertificate,
    shard_secret: ShardSecret,
    manifest: Manifest,
    modality: Modality,
}

impl ShardInputV1 {
    pub(crate) fn new(
        transaction_certificate: TransactionCertificate,
        shard_secret: ShardSecret,
        manifest: Manifest,
        modality: Modality,
    ) -> Self {
        Self {
            transaction_certificate,
            shard_secret,
            manifest,
            modality,
        }
    }
}

impl ShardInputAPI for ShardInputV1 {
    fn transaction_certificate(&self) -> &TransactionCertificate {
        &self.transaction_certificate
    }
    fn shard_secret(&self) -> &ShardSecret {
        &self.shard_secret
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn modality(&self) -> &Modality {
        &self.modality
    }
}

#[derive(Deserialize, Serialize)]
pub struct SignedShardInput {
    shard_input: ShardInput,
    signature: Bytes,
}

#[derive(Serialize, Deserialize)]
struct InnerShardInputDigest([u8; DIGEST_LENGTH]);

fn compute_inner_shard_input_digest(
    shard_input: &ShardInput,
) -> ShardResult<InnerShardInputDigest> {
    let mut hasher = DefaultHashFunction::new();
    hasher.update(bcs::to_bytes(shard_input).map_err(ShardError::SerializationFailure)?);
    Ok(InnerShardInputDigest(hasher.finalize().into()))
}

fn to_shard_input_scoped_message(
    digest: InnerShardInputDigest,
) -> ScopedMessage<InnerShardInputDigest> {
    ScopedMessage::new(Scope::ShardInput, digest)
}

fn compute_shard_input_signature(
    shard_input: &ShardInput,
    protocol_keypair: &ProtocolKeyPair,
) -> ShardResult<ProtocolKeySignature> {
    let digest = compute_inner_shard_input_digest(shard_input)?;
    let message = bcs::to_bytes(&to_shard_input_scoped_message(digest))
        .map_err(ShardError::SerializationFailure)?;
    Ok(protocol_keypair.sign(&message))
}
fn verify_shard_input_signature(
    shard_input: &ShardInput,
    signature: &[u8],
    protocol_pubkey: &ProtocolPublicKey,
) -> ShardResult<()> {
    let digest = compute_inner_shard_input_digest(shard_input)?;
    let message = bcs::to_bytes(&to_shard_input_scoped_message(digest))
        .map_err(ShardError::SerializationFailure)?;
    let sig =
        ProtocolKeySignature::from_bytes(signature).map_err(ShardError::MalformedSignature)?;
    protocol_pubkey
        .verify(&message, &sig)
        .map_err(ShardError::SignatureVerificationFailure)
}

impl Deref for SignedShardInput {
    type Target = ShardInput;

    fn deref(&self) -> &Self::Target {
        &self.shard_input
    }
}

#[derive(Clone)]
pub struct VerifiedShardInput {
    block: Arc<SignedShardInput>,
    // add digest or request
    serialized: Bytes,
}

impl VerifiedShardInput {
    pub(crate) fn new(signed_shard_input: SignedShardInput, serialized: Bytes) -> Self {
        Self {
            block: Arc::new(signed_shard_input),
            serialized,
        }
    }
}

/// Digest of a `ShardInput` which covers the `ShardInput` in Bytes format.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShardInputDigest([u8; DIGEST_LENGTH]);

impl ShardInputDigest {
    /// Lexicographic min & max digest.
    pub const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    pub const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl Hash for ShardInputDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<ShardInputDigest> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: ShardInputDigest) -> Self {
        Digest::new(hd.0)
    }
}

impl fmt::Display for ShardInputDigest {
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

impl fmt::Debug for ShardInputDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for ShardInputDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
