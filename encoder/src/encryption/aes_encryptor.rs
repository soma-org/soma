use crate::{crypto::AesKey, error::ShardResult};
use aes::cipher::StreamCipher;
use aes::{cipher::BlockEncrypt, Aes256};
use bytes::{Bytes, BytesMut};
use crypto_common::KeyIvInit;

use crate::encryption::Encryptor;

type Aes256Ctr64LE = ctr::Ctr64LE<aes::Aes256>;

pub(crate) struct Aes256Ctr64LEEncryptor {}

impl Aes256Ctr64LEEncryptor {
    const NONCE: [u8; 16] = [0u8; 16];

    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Encryptor<AesKey> for Aes256Ctr64LEEncryptor {
    fn encrypt(&self, key: AesKey, contents: Bytes) -> Bytes {
        let mut cipher = Aes256Ctr64LE::new(&key, &Self::NONCE.into());
        let mut buffer = BytesMut::with_capacity(contents.len());
        buffer.extend_from_slice(&contents);
        cipher.apply_keystream(&mut buffer);
        buffer.freeze()
    }

    fn decrypt(&self, key: AesKey, contents: Bytes) -> Bytes {
        // CTR mode is symmetric, so encryption and decryption are the same
        self.encrypt(key, contents)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use arbtest::{arbitrary, arbtest};
    use bytes::Bytes;

    #[test]
    fn encryption_decryption_roundtrip() {
        // property function: generate random key & contents, check that
        // decrypt(encrypt(contents)) == contents
        arbtest(|u| {
            // Generate a random 256-bit key
            let key_bytes: [u8; 32] = u.arbitrary()?;
            let aes_key = AesKey::from(key_bytes);

            // Generate random contents
            let contents: Vec<u8> = u.arbitrary()?;

            let encryptor = Aes256Ctr64LEEncryptor::new();

            // Encrypt
            let encrypted = encryptor.encrypt(aes_key.clone(), Bytes::from(contents.clone()));
            // Decrypt
            let decrypted = encryptor.decrypt(aes_key, encrypted.clone());

            // Assert that the roundtrip is identical
            assert_eq!(decrypted.as_ref(), contents.as_slice());

            // Assert that encrypted does not match original contents as long as
            // contents is not an empty set.
            if contents.len() > 0 {
                assert_ne!(encrypted.as_ref(), contents.as_slice());
            }

            Ok(())
        });
    }
}
