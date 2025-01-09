#![doc = include_str!("README.md")]

use bytes::Bytes;

pub(crate) mod aes_encryptor;

pub(crate) trait Encryptor<T>: Send + Sync + Sized + 'static {
    fn encrypt(&self, key: T, contents: Bytes) -> Bytes;
    fn decrypt(&self, key: T, contents: Bytes) -> Bytes;
}
