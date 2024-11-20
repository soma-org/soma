use crate::{crypto::AesKey, error::ShardResult};
use aes::cipher::StreamCipher;
use aes::{cipher::BlockEncrypt, Aes256};
use bytes::{Bytes, BytesMut};
use crypto_common::KeyIvInit;

use super::BlobEncryption;

type Aes256Ctr64LE = ctr::Ctr64LE<aes::Aes256>;

pub(crate) struct AesEncryptor {}

impl AesEncryptor {
    const NONCE: [u8; 16] = [0u8; 16];

    fn new() -> Self {
        Self {}
    }
}

impl BlobEncryption<AesKey> for AesEncryptor {
    fn encrypt(key: AesKey, contents: Bytes) -> Bytes {
        let mut cipher = Aes256Ctr64LE::new(&key, &Self::NONCE.into());
        let mut buffer = BytesMut::with_capacity(contents.len());
        buffer.extend_from_slice(&contents);
        cipher.apply_keystream(&mut buffer);
        buffer.freeze()
    }

    fn decrypt(key: AesKey, contents: Bytes) -> Bytes {
        // CTR mode is symmetric, so encryption and decryption are the same
        Self::encrypt(key, contents)
    }
}
