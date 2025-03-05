use aes::cipher::StreamCipher;
use bytes::{Bytes, BytesMut};
use crypto_common::KeyIvInit;

use shared::{
    crypto::{EncryptionKey, Encryptor},
    error::SharedResult,
};

type Aes256Ctr64LE = ctr::Ctr64LE<aes::Aes256>;

pub(crate) struct Aes256Ctr64LEEncryptor {}

impl Aes256Ctr64LEEncryptor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Encryptor for Aes256Ctr64LEEncryptor {
    fn encrypt(&self, key: EncryptionKey, contents: Bytes) -> SharedResult<Bytes> {
        match key {
            EncryptionKey::Aes256(k) => {
                let mut cipher = Aes256Ctr64LE::new(&k.key, &k.iv.into());
                let mut buffer = BytesMut::with_capacity(contents.len());
                buffer.extend_from_slice(&contents);
                cipher.apply_keystream(&mut buffer);
                Ok(buffer.freeze())
            }
        }
    }

    fn decrypt(&self, key: EncryptionKey, contents: Bytes) -> SharedResult<Bytes> {
        // CTR mode is symmetric, so encryption and decryption are the same
        self.encrypt(key, contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arbtest::{arbitrary, arbtest};
    use bcs::from_bytes;
    use bytes::Bytes;
    use shared::crypto::{Aes256IV, Aes256Key};

    #[test]
    fn encryption_decryption_roundtrip() {
        // property function: generate random key & contents, check that
        // decrypt(encrypt(contents)) == contents
        arbtest(|u| {
            // Generate a random 256-bit key
            let key_bytes: [u8; 32] = u.arbitrary()?;
            let aes_key = Aes256Key::from(key_bytes);
            let aes_iv = Aes256IV {
                iv: [0u8; 16],
                key: aes_key,
            };

            let encryption_key = EncryptionKey::Aes256(aes_iv);

            // Generate random contents
            let contents: [u8; 32] = u.arbitrary()?;

            let encryptor = Aes256Ctr64LEEncryptor::new();

            // Encrypt
            let encrypted = encryptor
                .encrypt(encryption_key.clone(), Bytes::from(contents.to_vec()))
                .unwrap();
            // Decrypt
            let decrypted = encryptor
                .decrypt(encryption_key, encrypted.clone())
                .unwrap();

            // Assert that the roundtrip is identical
            assert_eq!(decrypted.as_ref(), contents.as_slice());

            // Assert that encrypted does not match original contents as long as
            // contents is not an empty set.
            if !contents.is_empty() {
                assert_ne!(encrypted.as_ref(), contents.as_slice());
            }
            Ok(())
        });
    }
}
