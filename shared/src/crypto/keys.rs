use fastcrypto::{
    bls12381, ed25519,
    error::FastCryptoError,
    traits::{AggregateAuthenticator, KeyPair as _, Signer as _, ToFromBytes, VerifyingKey as _},
};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Peer key is used for Peer and as the network identity of the authority.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PeerPublicKey(ed25519::Ed25519PublicKey);
pub struct PeerPrivateKey(ed25519::Ed25519PrivateKey);
pub struct PeerKeyPair(ed25519::Ed25519KeyPair);

impl PeerPublicKey {
    pub fn new(key: ed25519::Ed25519PublicKey) -> Self {
        Self(key)
    }

    pub fn into_inner(self) -> ed25519::Ed25519PublicKey {
        self.0
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.0 .0.to_bytes()
    }
}

impl PeerPrivateKey {
    pub fn into_inner(self) -> ed25519::Ed25519PrivateKey {
        self.0
    }
}

impl PeerKeyPair {
    pub fn new(keypair: ed25519::Ed25519KeyPair) -> Self {
        Self(keypair)
    }

    pub fn generate<R: rand::Rng + fastcrypto::traits::AllowedRng>(rng: &mut R) -> Self {
        Self(ed25519::Ed25519KeyPair::generate(rng))
    }

    pub fn public(&self) -> PeerPublicKey {
        PeerPublicKey(self.0.public().clone())
    }

    pub fn private_key(self) -> PeerPrivateKey {
        PeerPrivateKey(self.0.copy().private())
    }

    pub fn private_key_bytes(self) -> [u8; 32] {
        self.0.private().0.to_bytes()
    }
}

impl Clone for PeerKeyPair {
    fn clone(&self) -> Self {
        Self(self.0.copy())
    }
}

/// Protocol key is used for signing blocks and verifying block signatures.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ProtocolPublicKey(ed25519::Ed25519PublicKey);
pub struct ProtocolKeyPair(ed25519::Ed25519KeyPair);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProtocolKeySignature(ed25519::Ed25519Signature);

impl ProtocolPublicKey {
    pub fn new(key: ed25519::Ed25519PublicKey) -> Self {
        Self(key)
    }

    pub fn verify(
        &self,
        message: &[u8],
        signature: &ProtocolKeySignature,
    ) -> Result<(), FastCryptoError> {
        self.0.verify(message, &signature.0)
    }

    pub fn to_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl ProtocolKeyPair {
    pub fn new(keypair: ed25519::Ed25519KeyPair) -> Self {
        Self(keypair)
    }

    pub fn generate<R: rand::Rng + fastcrypto::traits::AllowedRng>(rng: &mut R) -> Self {
        Self(ed25519::Ed25519KeyPair::generate(rng))
    }

    pub fn public(&self) -> ProtocolPublicKey {
        ProtocolPublicKey(self.0.public().clone())
    }

    pub fn sign(&self, message: &[u8]) -> ProtocolKeySignature {
        ProtocolKeySignature(self.0.sign(message))
    }
}

impl Clone for ProtocolKeyPair {
    fn clone(&self) -> Self {
        Self(self.0.copy())
    }
}

impl ProtocolKeySignature {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        Ok(Self(ed25519::Ed25519Signature::from_bytes(bytes)?))
    }

    pub fn to_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl Hash for ProtocolKeySignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.as_bytes()[..8]);
    }
}

// /////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////

/// A BLS public key wrapper for encoding operations
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AuthorityPublicKey(bls12381::min_sig::BLS12381PublicKey);

/// A BLS keypair wrapper for encoding operations
#[derive(Debug)]
pub struct AuthorityKeyPair(bls12381::min_sig::BLS12381KeyPair);

/// A BLS signature wrapper for encoding operations
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthoritySignature(bls12381::min_sig::BLS12381Signature);

/// A BLS aggregate signature wrapper for encoding operations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthorityAggregateSignature(bls12381::min_sig::BLS12381AggregateSignature);

impl AuthorityPublicKey {
    /// Creates a new `AuthorityPublicKey` from a BLS12381PublicKey
    pub fn new(key: bls12381::min_sig::BLS12381PublicKey) -> Self {
        Self(key)
    }

    /// Returns a reference to the inner BLS12381PublicKey
    pub fn inner(&self) -> &bls12381::min_sig::BLS12381PublicKey {
        &self.0
    }

    /// Returns the public key as a byte slice
    pub fn to_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    /// Verifies a signature against a message
    pub fn verify(
        &self,
        msg: &[u8],
        signature: &AuthoritySignature,
    ) -> Result<(), FastCryptoError> {
        self.0.verify(msg, &signature.0)
    }
}

impl AuthorityKeyPair {
    /// Creates a new `AuthorityKeyPair` from a BLS12381KeyPair
    pub fn new(keypair: bls12381::min_sig::BLS12381KeyPair) -> Self {
        Self(keypair)
    }

    /// Generates a new random keypair using the provided RNG
    pub fn generate<R: rand::Rng + fastcrypto::traits::AllowedRng>(rng: &mut R) -> Self {
        Self(bls12381::min_sig::BLS12381KeyPair::generate(rng))
    }

    /// Returns the public key associated with this keypair
    pub fn public(&self) -> AuthorityPublicKey {
        AuthorityPublicKey(self.0.public().clone())
    }

    /// Signs a message using this keypair
    pub fn sign(&self, msg: &[u8]) -> AuthoritySignature {
        AuthoritySignature(self.0.sign(msg))
    }
}

impl Clone for AuthorityKeyPair {
    fn clone(&self) -> Self {
        Self(self.0.copy())
    }
}

impl AuthoritySignature {
    /// Creates a new `AuthoritySignature` from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        bls12381::min_sig::BLS12381Signature::from_bytes(bytes).map(AuthoritySignature)
    }

    /// Returns the signature as bytes
    pub fn to_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl AuthorityAggregateSignature {
    /// Creates a new aggregate signature from a slice of signatures
    pub fn new(signatures: &[AuthoritySignature]) -> Result<Self, FastCryptoError> {
        Ok(Self(
            bls12381::min_sig::BLS12381AggregateSignature::aggregate(
                signatures.iter().map(|sig| &sig.0),
            )?,
        ))
    }

    /// Verifies the aggregate signature against multiple public keys and a message
    pub fn verify(
        &self,
        pks: &[AuthorityPublicKey],
        message: &[u8],
    ) -> Result<(), FastCryptoError> {
        let inner_keys: Vec<_> = pks.iter().map(|pk| pk.inner().to_owned()).collect();
        self.0.verify(&inner_keys, message)
    }
}

// ///////////////////////////////////////////////////////////////////////////
// document + test
// ///////////////////////////////////////////////////////////////////////////

/// A BLS public key wrapper for encoding operations
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub struct EncoderPublicKey(bls12381::min_sig::BLS12381PublicKey);

/// A BLS keypair wrapper for encoding operations
#[derive(Debug)]
pub struct EncoderKeyPair(bls12381::min_sig::BLS12381KeyPair);

/// A BLS signature wrapper for encoding operations
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderSignature(bls12381::min_sig::BLS12381Signature);

/// A BLS aggregate signature wrapper for encoding operations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncoderAggregateSignature(bls12381::min_sig::BLS12381AggregateSignature);

impl EncoderPublicKey {
    /// Creates a new `EncoderPublicKey` from a BLS12381PublicKey
    pub fn new(key: bls12381::min_sig::BLS12381PublicKey) -> Self {
        Self(key)
    }

    /// Returns a reference to the inner BLS12381PublicKey
    pub fn inner(&self) -> &bls12381::min_sig::BLS12381PublicKey {
        &self.0
    }

    /// Returns the public key as a byte slice
    pub fn to_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    /// Verifies a signature against a message
    pub fn verify(&self, msg: &[u8], signature: &EncoderSignature) -> Result<(), FastCryptoError> {
        self.0.verify(msg, &signature.0)
    }
}

impl EncoderKeyPair {
    /// Creates a new `EncoderKeyPair` from a BLS12381KeyPair
    pub fn new(keypair: bls12381::min_sig::BLS12381KeyPair) -> Self {
        Self(keypair)
    }

    /// Generates a new random keypair using the provided RNG
    pub fn generate<R: rand::Rng + fastcrypto::traits::AllowedRng>(rng: &mut R) -> Self {
        Self(bls12381::min_sig::BLS12381KeyPair::generate(rng))
    }
    pub fn inner(&self) -> &bls12381::min_sig::BLS12381KeyPair {
        &self.0
    }

    /// Returns the public key associated with this keypair
    pub fn public(&self) -> EncoderPublicKey {
        EncoderPublicKey(self.0.public().clone())
    }

    /// Signs a message using this keypair
    pub fn sign(&self, msg: &[u8]) -> EncoderSignature {
        EncoderSignature(self.0.sign(msg))
    }
}

impl Clone for EncoderKeyPair {
    fn clone(&self) -> Self {
        Self(self.0.copy())
    }
}

impl EncoderSignature {
    /// Creates a new `EncoderSignature` from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        bls12381::min_sig::BLS12381Signature::from_bytes(bytes).map(EncoderSignature)
    }

    /// Returns the signature as bytes
    pub fn to_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl EncoderAggregateSignature {
    /// Creates a new aggregate signature from a slice of signatures
    pub fn new(signatures: &[EncoderSignature]) -> Result<Self, FastCryptoError> {
        Ok(Self(
            bls12381::min_sig::BLS12381AggregateSignature::aggregate(
                signatures.iter().map(|sig| &sig.0),
            )?,
        ))
    }

    /// Verifies the aggregate signature against multiple public keys and a message
    pub fn verify(&self, pks: &[EncoderPublicKey], message: &[u8]) -> Result<(), FastCryptoError> {
        let inner_keys: Vec<_> = pks.iter().map(|pk| pk.inner().to_owned()).collect();
        self.0.verify(&inner_keys, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_single_signature_verification() {
        let message = b"Hello, world!";
        let mut rng = thread_rng();

        // Generate a keypair and sign message
        let keypair = EncoderKeyPair::generate(&mut rng);
        let signature = keypair.sign(message);

        // Verify the signature
        let public_key = keypair.public();
        assert!(public_key.verify(message, &signature).is_ok());

        // Test with wrong message
        let wrong_message = b"Wrong message";
        assert!(public_key.verify(wrong_message, &signature).is_err());
    }

    #[test]
    fn test_aggregate_signature_verification() {
        let message = b"Hello, world!";
        let mut rng = thread_rng();

        // Generate two keypairs and signatures
        let keypair1 = EncoderKeyPair::generate(&mut rng);
        let signature1 = keypair1.sign(message);

        let keypair2 = EncoderKeyPair::generate(&mut rng);
        let signature2 = keypair2.sign(message);

        // Create aggregate signature
        let signatures = vec![signature1, signature2];
        let aggregate_signature = EncoderAggregateSignature::new(&signatures).unwrap();

        // Verify aggregate signature
        let public_keys = vec![keypair1.public(), keypair2.public()];
        assert!(aggregate_signature.verify(&public_keys, message).is_ok());

        // Test with wrong message
        let wrong_message = b"Wrong message";
        assert!(aggregate_signature
            .verify(&public_keys, wrong_message)
            .is_err());
    }
}
