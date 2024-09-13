use crate::crypto::{
    keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
    DefaultHashFunction, DIGEST_LENGTH,
};
use crate::error::{ShardError, ShardResult};
use crate::types::scope::{Scope, ScopedMessage};
use fastcrypto::hash::HashFunction;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct MacroTest {
    f1: u64,
}

/// `SignedMacroTest` contains a `MacroTest` instance and its corresponding signature.
///
/// The signature is computed by serializing the `MacroTest` instance and hashing it
/// to produce an inner digest used solely for signing. This inner digest is then
/// wrapped in a `ScopedMessage` with a `Scope` that matches the struct name.
/// Scopes help prevent malicious signature reuse across different domains.
/// The resulting scoped message is then signed.
///
/// Note: The recommended way to refer to a signed type is by computing a digest that
/// includes the signature. This ensures that different valid signatures for the same
/// content result in different digests.
#[derive(Debug, Deserialize, Serialize)]
pub struct SignedMacroTest {
    /// The underlying `MacroTest` instance
    inner: MacroTest,
    /// The byte representation of the signature
    signature: bytes::Bytes,
}

/// A private type representing the digest of `MacroTest`.
#[derive(Serialize, Deserialize)]
struct InnerMacroTestDigest([u8; DIGEST_LENGTH]);

impl std::ops::Deref for SignedMacroTest {
    type Target = MacroTest;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl MacroTest {
    /// Signs the `MacroTest` instance using the provided keypair.
    ///
    /// This method internally calculates an inner digest, scopes the message,
    /// and derives the signature.
    ///
    /// # Arguments
    ///
    /// * `keypair` - The `ProtocolKeyPair` used for signing.
    ///
    /// # Returns
    ///
    /// A `ShardResult` containing the `SignedMacroTest` if successful.
    pub fn sign(self, keypair: &ProtocolKeyPair) -> ShardResult<SignedMacroTest> {
        let signature = self.compute_signature(keypair)?;
        Ok(SignedMacroTest {
            inner: self,
            signature: bytes::Bytes::copy_from_slice(signature.to_bytes()),
        })
    }

    /// Computes the inner digest of the `MacroTest` instance.
    ///
    /// This method serializes the `MacroTest` using `bcs`, then hashes the result.
    fn inner_digest(&self) -> ShardResult<InnerMacroTestDigest> {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(bcs::to_bytes(self).map_err(ShardError::SerializationFailure)?);
        Ok(InnerMacroTestDigest(hasher.finalize().into()))
    }

    /// Creates a `ScopedMessage` with `Scope::MacroTest` for the given digest.
    const fn scoped_message(digest: InnerMacroTestDigest) -> ScopedMessage<InnerMacroTestDigest> {
        ScopedMessage::new(Scope::MacroTest, digest)
    }

    /// Computes the signature for the `MacroTest` instance.
    ///
    /// This method calls `inner_digest`, `scoped_message`, and then signs the resulting message.
    fn compute_signature(&self, keypair: &ProtocolKeyPair) -> ShardResult<ProtocolKeySignature> {
        let digest = self.inner_digest()?;
        let message = bcs::to_bytes(&Self::scoped_message(digest))
            .map_err(ShardError::SerializationFailure)?;
        Ok(keypair.sign(&message))
    }
}

impl SignedMacroTest {
    /// Verifies the signature of the `SignedMacroTest` instance.
    ///
    /// This method computes a hash digest of the inner `MacroTest`, converts it to a
    /// `ScopedMessage` with `Scope::MacroTest`, and verifies the signature against
    /// the provided public key.
    ///
    /// # Arguments
    ///
    /// * `public_key` - The `ProtocolPublicKey` used for verification.
    ///
    /// # Returns
    ///
    /// A `ShardResult` indicating success or failure of the verification.
    pub fn verify_signature(&self, public_key: &ProtocolPublicKey) -> ShardResult<()> {
        let inner = &self.inner;
        let digest = inner.inner_digest()?;
        let message = bcs::to_bytes(&MacroTest::scoped_message(digest))
            .map_err(ShardError::SerializationFailure)?;
        let sig = ProtocolKeySignature::from_bytes(&self.signature)
            .map_err(ShardError::MalformedSignature)?;
        public_key
            .verify(&message, &sig)
            .map_err(ShardError::SignatureVerificationFailure)?;

        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////
// DIGEST
////////////////////////////////////////////////////////////////////////

/// Represents a hash digest for `SignedMacroTest`.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct SignedMacroTestDigest([u8; DIGEST_LENGTH]);

impl SignedMacroTestDigest {
    /// Lexicographically minimal digest
    const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    /// Lexicographically maximal digest
    const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}
impl std::hash::Hash for SignedMacroTestDigest {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<SignedMacroTestDigest> for fastcrypto::hash::Digest<{ DIGEST_LENGTH }> {
    fn from(hd: SignedMacroTestDigest) -> Self {
        fastcrypto::hash::Digest::new(hd.0)
    }
}

impl std::fmt::Display for SignedMacroTestDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
                .get(0..4)
                .ok_or(std::fmt::Error)?
        )
    }
}

impl std::fmt::Debug for SignedMacroTestDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for SignedMacroTestDigest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl SignedMacroTest {
    /// Serializes the `SignedMacroTest` instance using `bcs`.
    ///
    /// # Returns
    ///
    /// A `ShardResult` containing the serialized `bytes::Bytes` if successful.
    pub fn serialize(&self) -> ShardResult<bytes::Bytes> {
        let bytes = bcs::to_bytes(self).map_err(ShardError::SerializationFailure)?;
        Ok(bytes.into())
    }

    /// Computes the digest of the serialized `SignedMacroTest` instance.
    ///
    /// # Arguments
    ///
    /// * `serialized` - The serialized `bytes::Bytes` of the `SignedMacroTest`.
    ///
    /// # Returns
    ///
    /// The computed `SignedMacroTestDigest`.
    fn compute_digest_from_serialized(&self, serialized: &bytes::Bytes) -> SignedMacroTestDigest {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(serialized);
        SignedMacroTestDigest(hasher.finalize().into())
    }
}

////////////////////////////////////////////////////////////////////////
// VERIFICATION
////////////////////////////////////////////////////////////////////////

/// Represents a verified `SignedMacroTest` instance.
///
/// This struct holds a verified `SignedMacroTest` along with its cached digest
/// and serialized form for efficiency. The underlying data is refcounted,
/// making `clone()` operations relatively inexpensive.
#[derive(Clone)]
pub struct VerifiedSignedMacroTest {
    /// The verified `SignedMacroTest` instance
    inner: std::sync::Arc<SignedMacroTest>,
    /// The cached digest to avoid recomputation
    digest: SignedMacroTestDigest,
    /// The cached serialized bytes to avoid recomputation
    serialized: bytes::Bytes,
}

impl VerifiedSignedMacroTest {
    /// Returns the cached digest of the verified `SignedMacroTest`.
    pub(crate) const fn digest(&self) -> SignedMacroTestDigest {
        self.digest
    }

    /// Returns the cached serialized bytes of the verified `SignedMacroTest`.
    pub(crate) const fn serialized(&self) -> &bytes::Bytes {
        &self.serialized
    }
}

impl std::ops::Deref for VerifiedSignedMacroTest {
    type Target = SignedMacroTest;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl PartialEq for VerifiedSignedMacroTest {
    fn eq(&self, other: &Self) -> bool {
        self.digest() == other.digest()
    }
}

impl SignedMacroTest {
    /// Applies a custom check to the `SignedMacroTest` instance.
    ///
    /// This method allows for arbitrary verification logic to be applied
    /// to a `SignedMacroTest`.
    ///
    /// # Arguments
    ///
    /// * `closure` - A closure that performs the custom check.
    ///
    /// # Returns
    ///
    /// A `ShardResult` indicating success or failure of the check.
    pub fn check<F>(&self, closure: F) -> ShardResult<()>
    where
        F: FnOnce(&Self) -> ShardResult<()>,
    {
        closure(self)
    }

    /// Verifies the `SignedMacroTest` instance and creates a `VerifiedSignedMacroTest`.
    ///
    /// This method applies a custom verification closure to the `SignedMacroTest`.
    /// If the verification succeeds, it constructs and returns a `VerifiedSignedMacroTest`.
    ///
    /// # Arguments
    ///
    /// * `closure` - A closure that performs the custom verification.
    ///
    /// # Returns
    ///
    /// A `ShardResult` containing the `VerifiedSignedMacroTest` if verification succeeds.
    pub fn verify<F>(self, closure: F) -> ShardResult<VerifiedSignedMacroTest>
    where
        F: FnOnce(&Self) -> ShardResult<()>,
    {
        self.check(closure)?;

        let serialized = self.serialize()?;
        let digest = self.compute_digest_from_serialized(&serialized);

        Ok(VerifiedSignedMacroTest {
            inner: std::sync::Arc::new(self),
            digest,
            serialized,
        })
    }
}
