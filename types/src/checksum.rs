use fastcrypto::{
    error::FastCryptoError,
    hash::{Digest, HashFunction},
    serialize_deserialize_with_to_from_bytes,
    traits::{EncodeDecodeBase64, ToFromBytes},
};
use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
};

use crate::crypto::{DefaultHash as DefaultHashFunction, DIGEST_LENGTH};

/// Checksum is a bytes checksum for data. We use the same default hash function
/// as the rest of the network. There are associated functions for new from bytes
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Checksum(pub [u8; DIGEST_LENGTH]);

impl Checksum {
    pub(crate) const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    pub(crate) const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);

    pub fn new_from_hash(hash: [u8; DIGEST_LENGTH]) -> Self {
        Self(hash)
    }

    // TODO: make this work better for chunking intelligently
    pub fn new_from_bytes(bytes: &[u8]) -> Self {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(bytes);
        Self(hasher.finalize().into())
    }

    // TODO: make
}

impl Hash for Checksum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<Checksum> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: Checksum) -> Self {
        Digest::new(hd.0)
    }
}

impl From<Checksum> for [u8; 32] {
    fn from(checksum: Checksum) -> Self {
        let mut seed = [0u8; 32];
        match DIGEST_LENGTH.cmp(&32) {
            Ordering::Equal => seed.copy_from_slice(&checksum.0),
            Ordering::Greater => seed.copy_from_slice(&checksum.0[..32]),
            Ordering::Less => {
                for (i, byte) in checksum.0.iter().cycle().take(32).enumerate() {
                    seed[i] = *byte;
                }
            }
        }
        seed
    }
}

impl fmt::Display for Checksum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE, self.0)
        )
    }
}

impl fmt::Debug for Checksum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl AsRef<[u8]> for Checksum {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl ToFromBytes for Checksum {
    fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        if bytes.len() != DIGEST_LENGTH {
            return Err(FastCryptoError::InvalidInput);
        }
        let mut arr = [0u8; DIGEST_LENGTH];
        arr.copy_from_slice(bytes);
        Ok(Self(arr))
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

serialize_deserialize_with_to_from_bytes!(Checksum, DIGEST_LENGTH);
