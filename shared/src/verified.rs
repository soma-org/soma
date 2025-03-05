use crate::{
    crypto::keys::ProtocolKeyPair,
    digest::Digest,
    error::{SharedError, SharedResult},
    scope::Scope,
};
use bytes::{Buf, Bytes};
use serde::Serialize;
use std::sync::Arc;

use super::{serialized::Serialized, signed::Signed};

#[derive(Debug)]
pub struct Verified<T: Serialize> {
    inner: Arc<T>,
    digest: Digest<T>,
    serialized: Bytes,
}

impl<T: Serialize> Verified<T> {
    pub fn new<F>(inner: T, serialized: Bytes, verifier: F) -> SharedResult<Self>
    where
        F: FnOnce(&T) -> SharedResult<()>,
    {
        verifier(&inner)?;
        let digest = Digest::<T>::new_from_bytes(&serialized);

        Ok(Self {
            inner: Arc::new(inner),
            digest,
            serialized,
        })
    }

    pub fn from_trusted(inner: T) -> SharedResult<Self> {
        let serialized = Bytes::copy_from_slice(
            &bcs::to_bytes(&inner).map_err(SharedError::SerializationFailure)?,
        );

        Ok(Self {
            inner: Arc::new(inner),
            digest: Digest::<T>::new_from_bytes(&serialized),
            serialized,
        })
    }

    pub fn from_trusted_bytes(inner: T, serialized: Bytes) -> SharedResult<Self> {
        let digest = Digest::<T>::new_from_bytes(&serialized);
        Ok(Self {
            inner: Arc::new(inner),
            digest,
            serialized,
        })
    }

    pub fn digest(&self) -> Digest<T> {
        self.digest
    }

    pub fn serialized(&self) -> Serialized<T> {
        Serialized::new(self.serialized.clone())
    }

    pub fn bytes(&self) -> Bytes {
        self.serialized.clone()
    }
}

impl<T: Serialize> std::ops::Deref for Verified<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: Serialize> PartialEq for Verified<T> {
    fn eq(&self, other: &Self) -> bool {
        self.digest() == other.digest()
    }
}

impl<T: Serialize> Clone for Verified<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            digest: self.digest,
            serialized: self.serialized.clone(),
        }
    }
}
