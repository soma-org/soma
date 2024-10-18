use bytes::Bytes;

use crate::{
    crypto::{DefaultHashFunction, DIGEST_LENGTH},
    error::{ShardError, ShardResult},
    // error::{ShardError, ShardResult},
};
use std::marker::PhantomData;

pub(crate) struct Digest<T> {
    inner: [u8; DIGEST_LENGTH],
    marker: PhantomData<T>,
}

impl<T> Digest<T> {
    pub fn new(inner: &T) -> ShardResult<Self> {
        let serialized_inner = bcs::to_bytes(inner).map_err(ShardError::SerializationFailure)?;
        let mut hasher = DefaultHashFunction::new();
        hasher.update(serialized_inner);
        Ok(Self {
            inner: hasher.finalize.into(),
            marker: PhantomData,
        })
    }
    pub fn new_from_bytes(bytes: &Bytes) -> Self {
        let mut hasher = DefaultHashFunction::new();
        hasher.update(bytes);
        Self {
            inner: hasher.finalize.into(),
            marker: PhantomData,
        }
    }
}

impl<T> Digest<T> {
    /// Lexicographically minimal digest
    pub const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    /// Lexicographically maximal digest
    pub const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);
}

impl<T> std::hash::Hash for Digest<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl<T> From<Digest<T>> for fastcrypto::hash::Digest<{ DIGEST_LENGTH }> {
    fn from(hd: Digest<T>) -> Self {
        fastcrypto::hash::Digest::new(hd.0)
    }
}

impl<T> std::fmt::Display for Digest<T> {
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

impl<T> std::fmt::Debug for Digest<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

impl<T> AsRef<[u8]> for Digest<T> {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
