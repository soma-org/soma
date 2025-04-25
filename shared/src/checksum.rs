use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    hash::{Hash, Hasher},
};

use crate::crypto::{DefaultHashFunction, DIGEST_LENGTH};

/// Checksum is a bytes checksum for data. We use the same default hash function
/// as the rest of the network. There are associated functions for new from bytes
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
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

impl fmt::Display for Checksum {
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
