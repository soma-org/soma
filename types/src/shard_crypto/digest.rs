use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    crypto::{DefaultHash as DefaultHashFunction, DIGEST_LENGTH},
    error::{SharedError, SharedResult},
};
use fastcrypto::hash::HashFunction;
use std::{cmp::Ordering, marker::PhantomData};

#[derive(Serialize, Deserialize)]
pub struct Digest<T: Serialize> {
    pub inner: [u8; DIGEST_LENGTH],
    pub marker: PhantomData<T>,
}
impl<T: Serialize> Clone for Digest<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Serialize> Copy for Digest<T> {}

impl<T: Serialize> PartialEq for Digest<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl<T: Serialize> Eq for Digest<T> {}

impl<T: Serialize> Ord for Digest<T> {
    fn cmp(&self, other: &Digest<T>) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}

impl<T: Serialize> PartialOrd for Digest<T> {
    fn partial_cmp(&self, other: &Digest<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Serialize> Digest<T> {
    pub fn new(inner: &T) -> SharedResult<Self> {
        let serialized_inner = bcs::to_bytes(inner).map_err(SharedError::SerializationFailure)?;
        let mut hasher = DefaultHashFunction::new();
        hasher.update(serialized_inner);
        Ok(Self {
            inner: hasher.finalize().into(),
            marker: PhantomData,
        })
    }
    pub fn new_from_bytes<Data: AsRef<[u8]>>(data: Data) -> Self {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(data);
        Self {
            inner: hasher.finalize().into(),
            marker: PhantomData,
        }
    }

    pub fn from_raw(bytes: [u8; DIGEST_LENGTH]) -> Self {
        Self {
            inner: bytes,
            marker: PhantomData,
        }
    }

    /// Extract the inner bytes array
    pub fn into_inner(self) -> [u8; DIGEST_LENGTH] {
        self.inner
    }
}

impl<T: Serialize> AsRef<[u8; DIGEST_LENGTH]> for Digest<T> {
    fn as_ref(&self) -> &[u8; DIGEST_LENGTH] {
        &self.inner
    }
}

impl<T: Serialize> From<[u8; DIGEST_LENGTH]> for Digest<T> {
    fn from(bytes: [u8; DIGEST_LENGTH]) -> Self {
        Self::from_raw(bytes)
    }
}

impl<T: Serialize> From<Digest<T>> for Vec<u8> {
    fn from(digest: Digest<T>) -> Self {
        digest.inner.to_vec()
    }
}

impl<T: Serialize> From<Digest<T>> for Bytes {
    fn from(digest: Digest<T>) -> Self {
        Bytes::copy_from_slice(&digest.inner)
    }
}

impl<T: Serialize> TryFrom<Vec<u8>> for Digest<T> {
    type Error = SharedError;

    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        let inner: [u8; DIGEST_LENGTH] = bytes
            .try_into()
            .map_err(|v: Vec<u8>| SharedError::InvalidDigestLength)?;
        Ok(Self::from_raw(inner))
    }
}

impl<T: Serialize> TryFrom<&[u8]> for Digest<T> {
    type Error = SharedError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let inner: [u8; DIGEST_LENGTH] = bytes
            .try_into()
            .map_err(|_| SharedError::InvalidDigestLength)?;
        Ok(Self::from_raw(inner))
    }
}

impl<T: Serialize> TryFrom<Bytes> for Digest<T> {
    type Error = SharedError;

    fn try_from(bytes: Bytes) -> Result<Self, Self::Error> {
        Self::try_from(bytes.as_ref())
    }
}

impl<T: Serialize> Digest<T> {
    /// Lexicographically minimal digest
    pub const MIN: Self = Self {
        inner: [u8::MIN; DIGEST_LENGTH],
        marker: PhantomData,
    };
    /// Lexicographically maximal digest
    pub const MAX: Self = Self {
        inner: [u8::MAX; DIGEST_LENGTH],
        marker: PhantomData,
    };
}

impl<T: Serialize> std::hash::Hash for Digest<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(&self.inner[..8]);
    }
}

impl<T: Serialize> From<Digest<T>> for fastcrypto::hash::Digest<{ DIGEST_LENGTH }> {
    fn from(hd: Digest<T>) -> Self {
        fastcrypto::hash::Digest::new(hd.inner)
    }
}

impl<T: Serialize> From<Digest<T>> for [u8; 32] {
    fn from(digest: Digest<T>) -> Self {
        let mut seed = [0u8; 32];
        match DIGEST_LENGTH.cmp(&32) {
            Ordering::Equal => seed.copy_from_slice(&digest.inner),
            Ordering::Greater => seed.copy_from_slice(&digest.inner[..32]),
            Ordering::Less => {
                for (i, byte) in digest.inner.iter().cycle().take(32).enumerate() {
                    seed[i] = *byte;
                }
            }
        }
        seed
    }
}

impl<T: Serialize> std::fmt::Display for Digest<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.inner) // .get(0..4)
                                                                                           // .ok_or(std::fmt::Error)?
        )
    }
}

impl<T: Serialize> std::fmt::Debug for Digest<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.inner)
        )
    }
}

impl<T: Serialize> AsRef<[u8]> for Digest<T> {
    fn as_ref(&self) -> &[u8] {
        &self.inner
    }
}

impl<T: Serialize> Default for Digest<T> {
    fn default() -> Self {
        Self {
            inner: [0u8; DIGEST_LENGTH],
            marker: PhantomData,
        }
    }
}
