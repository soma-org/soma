use crate::types::Address;
use crate::types::Digest;

use blake2::Digest as DigestTrait;

type Blake2b256 = blake2::Blake2b<blake2::digest::consts::U32>;

/// A Blake2b256 Hasher
#[derive(Debug, Default)]
pub struct Hasher(Blake2b256);

impl Hasher {
    /// Initialize a new Blake2b256 Hasher instance.
    pub fn new() -> Self {
        Self(Blake2b256::new())
    }

    /// Process the provided data, updating internal state.
    pub fn update<T: AsRef<[u8]>>(&mut self, data: T) {
        self.0.update(data)
    }

    /// Finalize hashing, consuming the Hasher instance and returning the resultant hash or
    /// `Digest`.
    pub fn finalize(self) -> Digest {
        let mut buf = [0; Digest::LENGTH];
        let result = self.0.finalize();

        buf.copy_from_slice(result.as_slice());

        Digest::new(buf)
    }

    /// Convenience function for creating a new Hasher instance, hashing the provided data, and
    /// returning the resultant `Digest`
    pub fn digest<T: AsRef<[u8]>>(data: T) -> Digest {
        let mut hasher = Self::new();
        hasher.update(data);
        hasher.finalize()
    }
}

impl std::io::Write for Hasher {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

impl crate::types::Ed25519PublicKey {
    /// Derive an `Address` from this Public Key
    ///
    /// An `Address` can be derived from an `Ed25519PublicKey` by hashing the bytes of the public
    /// key prefixed with the Ed25519 `SignatureScheme` flag (`0x00`).
    ///
    /// `hash( 0x00 || 32-byte ed25519 public key)`
    ///
    /// ```
    /// use sui_sdk_types::hash::Hasher;
    /// use sui_sdk_types::Address;
    /// use sui_sdk_types::Ed25519PublicKey;
    ///
    /// let public_key_bytes = [0; 32];
    /// let mut hasher = Hasher::new();
    /// hasher.update([0x00]); // The SignatureScheme flag for Ed25519 is `0`
    /// hasher.update(public_key_bytes);
    /// let address = Address::new(hasher.finalize().into_inner());
    /// println!("Address: {}", address);
    ///
    /// let public_key = Ed25519PublicKey::new(public_key_bytes);
    /// assert_eq!(address, public_key.derive_address());
    /// ```
    pub fn derive_address(&self) -> Address {
        let mut hasher = Hasher::new();
        self.write_into_hasher(&mut hasher);
        let digest = hasher.finalize();
        Address::new(digest.into_inner())
    }

    fn write_into_hasher(&self, hasher: &mut Hasher) {
        hasher.update([self.scheme().to_u8()]);
        hasher.update(self.inner());
    }
}

impl crate::types::Object {
    /// Calculate the digest of this `Object`
    ///
    /// This is done by hashing the BCS bytes of this `Object` prefixed
    pub fn digest(&self) -> Digest {
        const SALT: &str = "Object::";
        type_digest(SALT, self)
    }
}

impl crate::types::Transaction {
    pub fn digest(&self) -> Digest {
        const SALT: &str = "TransactionData::";
        type_digest(SALT, self)
    }
}

impl crate::types::TransactionEffects {
    pub fn digest(&self) -> Digest {
        const SALT: &str = "TransactionEffects::";
        type_digest(SALT, self)
    }
}

fn type_digest<T: serde::Serialize>(salt: &str, ty: &T) -> Digest {
    let mut hasher = Hasher::new();
    hasher.update(salt);
    bcs::serialize_into(&mut hasher, ty).unwrap();
    hasher.finalize()
}
