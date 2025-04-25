use bytes::Bytes;
use std::marker::PhantomData;

pub struct Serialized<T> {
    bytes: Bytes,
    marker: PhantomData<T>,
}

impl<T> Serialized<T> {
    pub fn new(bytes: Bytes) -> Self {
        Self {
            bytes,
            marker: PhantomData,
        }
    }
    pub fn bytes(&self) -> bytes::Bytes {
        self.bytes.clone()
    }
}

impl<T> std::ops::Deref for Serialized<T> {
    type Target = Bytes;

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}
