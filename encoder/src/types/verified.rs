use crate::{
    error::{ShardError, ShardResult},
    types::digest::Digest,
    ProtocolKeyPair, Scope,
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
    pub(crate) fn new<F>(inner: T, serialized: Bytes, verifier: F) -> ShardResult<Self>
    where
        F: FnOnce(&T) -> ShardResult<()>,
    {
        verifier(&inner)?;
        let digest = Digest::<T>::new_from_bytes(&serialized);

        Ok(Self {
            inner: Arc::new(inner),
            digest,
            serialized,
        })
    }

    pub(crate) fn from_trusted(inner: T) -> ShardResult<Self> {
        let serialized = Bytes::copy_from_slice(
            &bcs::to_bytes(&inner).map_err(ShardError::SerializationFailure)?,
        );

        Ok(Self {
            inner: Arc::new(inner),
            digest: Digest::<T>::new_from_bytes(&serialized),
            serialized,
        })
    }

    pub(crate) fn digest(&self) -> Digest<T> {
        self.digest.clone()
    }

    pub(crate) fn serialized(&self) -> Serialized<T> {
        Serialized::new(self.serialized.clone())
    }

    pub(crate) fn bytes(&self) -> Bytes {
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
            digest: self.digest.clone(),
            serialized: self.serialized.clone(),
        }
    }
}
