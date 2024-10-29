use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    crypto::{DefaultHashFunction, DIGEST_LENGTH},
    error::{ShardError, ShardResult},
    // error::{ShardError, ShardResult},
};
use fastcrypto::hash::HashFunction;
use std::marker::PhantomData;

#[derive(Serialize, Deserialize)]
pub(crate) struct Digest<T: Serialize> {
    inner: [u8; DIGEST_LENGTH],
    marker: PhantomData<T>,
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

impl<T: Serialize> Digest<T> {
    pub fn new(inner: &T) -> ShardResult<Self> {
        let serialized_inner = bcs::to_bytes(inner).map_err(ShardError::SerializationFailure)?;
        let mut hasher = DefaultHashFunction::new();
        hasher.update(serialized_inner);
        Ok(Self {
            inner: hasher.finalize().into(),
            marker: PhantomData,
        })
    }
    pub fn new_from_bytes(bytes: &Bytes) -> Self {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(bytes);
        Self {
            inner: hasher.finalize().into(),
            marker: PhantomData,
        }
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

impl<T: Serialize> std::fmt::Display for Digest<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.inner)
                .get(0..4)
                .ok_or(std::fmt::Error)?
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
