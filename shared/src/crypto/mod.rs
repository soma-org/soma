pub(crate) mod address;
pub mod keys;

use bytes::Bytes;
use crypto_common::generic_array::GenericArray;
use fastcrypto::hash::{Blake2b256, HashFunction};
use serde::Serialize;

pub type DefaultHashFunction = Blake2b256;
pub const DIGEST_LENGTH: usize = DefaultHashFunction::OUTPUT_SIZE;

pub type AesKey = GenericArray<u8, <aes::Aes256 as crypto_common::KeySizeUser>::KeySize>;
pub trait EncryptionKey: Serialize {}

impl EncryptionKey for AesKey {}

pub trait Encryptor<T>: Send + Sync + Sized + 'static {
    fn encrypt(&self, key: T, contents: Bytes) -> Bytes;
    fn decrypt(&self, key: T, contents: Bytes) -> Bytes;
}
