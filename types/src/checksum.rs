use fastcrypto::{error::FastCryptoError, hash::Digest, traits::ToFromBytes};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
};

use crate::crypto::{DIGEST_LENGTH, DefaultHash as DefaultHashFunction};

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
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, self.0)
        )
    }
}

impl fmt::Debug for Checksum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, self.0)
        )
    }
}

impl AsRef<[u8]> for Checksum {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Checksum {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // 1. Pull the raw string from the URL
        let s = String::deserialize(deserializer)?;

        // 2. Decode base64url (no padding) – exactly the format you use in `Display`
        let bytes = base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, &s)
            .map_err(|e| serde::de::Error::custom(format!("invalid base64url: {e}")))?;

        // 3. Length check – must be exactly DIGEST_LENGTH
        if bytes.len() != DIGEST_LENGTH {
            return Err(serde::de::Error::custom(format!(
                "checksum must be {DIGEST_LENGTH} bytes, got {}",
                bytes.len()
            )));
        }

        // 4. Copy into fixed-size array
        let mut arr = [0u8; DIGEST_LENGTH];
        arr.copy_from_slice(&bytes);
        Ok(Checksum(arr))
    }
}

impl Serialize for Checksum {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Encode the raw bytes as base64url (no padding) – exactly what `Display` does
        let encoded =
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, self.0);
        serializer.serialize_str(&encoded)
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
