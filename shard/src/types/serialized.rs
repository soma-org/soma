use bytes::Bytes;
use std::marker::PhantomData;

pub struct Serialized<T> {
    bytes: Bytes,
    marker: PhantomData<T>,
}

impl<T> Serialized<T> {
    pub(crate) fn new(bytes: Bytes) -> Self {
        Self {
            bytes,
            marker: PhantomData,
        }
    }
    pub(crate) const fn bytes(&self) -> &bytes::Bytes {
        &self.bytes
    }
}

impl<T> std::ops::Deref for Serialized<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
