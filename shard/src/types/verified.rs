use crate::{error::ShardResult, types::digest::Digest};
use bytes::Bytes;
use std::sync::Arc;

pub struct Verified<T> {
    inner: Arc<T>,
    digest: Digest<T>,
    serialized: Bytes,
}

impl<T> Verified<T> {
    pub(crate) fn new<F>(inner: T, serialized: Bytes, verifier: F) -> ShardResult<Self>
    where
        F: FnOnce(&T) -> ShardResult<()>,
    {
        verifier(&inner)?;
        let digest = Digest::<T>::new_from_bytes(&serialized);

        Ok(Self {
            inner: std::sync::Arc::new(inner),
            digest,
            serialized,
        })
    }

    pub(crate) const fn digest(&self) -> Digest<T> {
        self.digest
    }

    pub(crate) const fn serialized(&self) -> &bytes::Bytes {
        &self.serialized
    }
}

impl<T> std::ops::Deref for Verified<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> PartialEq for Verified<T> {
    fn eq(&self, other: &Self) -> bool {
        self.digest() == other.digest()
    }
}
