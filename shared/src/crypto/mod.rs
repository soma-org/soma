pub mod address;
pub mod keys;

use bytes::Bytes;
use crypto_common::generic_array::GenericArray;
use fastcrypto::hash::{Blake2b256, HashFunction};
use serde::{Deserialize, Serialize};

use crate::error::SharedResult;

pub type DefaultHashFunction = Blake2b256;
pub const DIGEST_LENGTH: usize = DefaultHashFunction::OUTPUT_SIZE;

pub type Aes256Key = GenericArray<u8, <aes::Aes256 as crypto_common::KeySizeUser>::KeySize>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Aes256IV {
    pub iv: [u8; 16],
    pub key: Aes256Key,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EncryptionKey {
    Aes256(Aes256IV),
}

impl Default for EncryptionKey {
    fn default() -> Self {
        EncryptionKey::Aes256(Aes256IV::default())
    }
}
pub trait EncryptorAPI: Send + Sync + Sized + 'static {
    fn encrypt(&self, key: EncryptionKey, contents: Bytes) -> SharedResult<Bytes>;
    fn decrypt(&self, key: EncryptionKey, contents: Bytes) -> SharedResult<Bytes>;
}
