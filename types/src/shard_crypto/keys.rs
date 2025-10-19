use fastcrypto::{
    bls12381::{self, min_sig::BLS12381PublicKey},
    ed25519::{self, Ed25519KeyPair},
    encoding::{Base64, Encoding},
    error::{FastCryptoError, FastCryptoResult},
    traits::{
        AggregateAuthenticator, EncodeDecodeBase64, KeyPair as _, Signer as _, ToFromBytes,
        VerifyingKey as _,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    str::FromStr,
};

/// A BLS public key wrapper for encoding operations
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub struct EncoderPublicKey(bls12381::min_sig::BLS12381PublicKey);

/// A BLS keypair wrapper for encoding operations
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
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

    pub fn MIN() -> Self {
        Self(BLS12381PublicKey::from_bytes(&[u8::MIN; BLS12381PublicKey::LENGTH]).unwrap())
    }
    pub fn MAX() -> Self {
        Self(BLS12381PublicKey::from_bytes(&[u8::MAX; BLS12381PublicKey::LENGTH]).unwrap())
    }
}

impl FromStr for EncoderPublicKey {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Handle both with and without 0x prefix
        let hex_str = s.trim_start_matches("0x");
        let bytes = hex::decode(hex_str).map_err(|e| format!("Invalid hex string: {}", e))?;

        Self::from_bytes(&bytes)
    }
}

impl EncoderPublicKey {
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        BLS12381PublicKey::from_bytes(bytes)
            .map(Self::new)
            .map_err(|e| format!("Invalid BLS public key: {}", e))
    }

    /// Convert to hex string with 0x prefix
    pub fn to_hex_string(&self) -> String {
        format!("0x{}", hex::encode(self.to_bytes()))
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

    /// Returns the signature as bytes
    pub fn to_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    /// Creates a new `EncoderAggregateSignature` from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        bls12381::min_sig::BLS12381AggregateSignature::from_bytes(bytes)
            .map(EncoderAggregateSignature)
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
