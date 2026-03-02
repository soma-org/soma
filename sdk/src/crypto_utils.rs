// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cryptographic utilities for the SOMA SDK.
//!
//! Encryption uses AES-256-CTR with a zero IV, matching the workspace standard
//! defined in `runtime/src/v1.rs`.

use types::config::genesis_config::SHANNONS_PER_SOMA;

// ---------------------------------------------------------------------------
// Unit conversion
// ---------------------------------------------------------------------------

/// Convert SOMA to shannons (the smallest on-chain unit).
pub fn to_shannons(soma: f64) -> u64 {
    (soma * SHANNONS_PER_SOMA as f64) as u64
}

/// Convert shannons to SOMA.
pub fn to_soma(shannons: u64) -> f64 {
    shannons as f64 / SHANNONS_PER_SOMA as f64
}

// ---------------------------------------------------------------------------
// Encryption: AES-256-CTR zero IV (matches runtime/src/v1.rs)
// ---------------------------------------------------------------------------

/// Encrypt data with AES-256-CTR (zero IV).
///
/// If `key` is `None`, a random 32-byte key is generated.
/// Returns `(encrypted_data, key)`.
pub fn encrypt_weights(data: &[u8], key: Option<&[u8; 32]>) -> (Vec<u8>, [u8; 32]) {
    use aes::Aes256;
    use ctr::Ctr128BE;
    use ctr::cipher::{KeyIvInit, StreamCipher};

    let key_bytes = match key {
        Some(k) => *k,
        None => {
            let mut k = [0u8; 32];
            rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut k);
            k
        }
    };

    let iv = [0u8; 16];
    let mut cipher = Ctr128BE::<Aes256>::new(&key_bytes.into(), &iv.into());
    let mut encrypted = data.to_vec();
    cipher.apply_keystream(&mut encrypted);

    (encrypted, key_bytes)
}

/// Decrypt data with AES-256-CTR (zero IV).
///
/// This is the same operation as encryption (CTR mode is symmetric).
/// Matches `runtime::v1::decrypt_aes256_ctr` exactly.
pub fn decrypt_weights(data: &[u8], key: &[u8; 32]) -> Vec<u8> {
    use aes::Aes256;
    use ctr::Ctr128BE;
    use ctr::cipher::{KeyIvInit, StreamCipher};

    let iv = [0u8; 16];
    let mut cipher = Ctr128BE::<Aes256>::new(key.into(), &iv.into());
    let mut decrypted = data.to_vec();
    cipher.apply_keystream(&mut decrypted);

    decrypted
}

// ---------------------------------------------------------------------------
// Hashing: Blake2b-256 (types::crypto::DefaultHash)
// ---------------------------------------------------------------------------

/// Compute the Blake2b-256 hash of `data`, returning a 32-byte array.
pub fn commitment(data: &[u8]) -> [u8; 32] {
    use fastcrypto::hash::HashFunction;
    let mut hasher = fastcrypto::hash::Blake2b256::default();
    hasher.update(data);
    let hash = hasher.finalize();
    let mut arr = [0u8; 32];
    arr.copy_from_slice(hash.as_ref());
    arr
}

/// Compute the Blake2b-256 hash of `data`, returning a Base58 string.
pub fn commitment_base58(data: &[u8]) -> String {
    use fastcrypto::encoding::{Base58, Encoding};
    Base58::encode(commitment(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_round_trip() {
        let data = b"hello world model weights";
        let (encrypted, key) = encrypt_weights(data, None);
        assert_ne!(&encrypted, data);
        let decrypted = decrypt_weights(&encrypted, &key);
        assert_eq!(&decrypted, data);
    }

    #[test]
    fn encrypt_with_fixed_key() {
        let data = b"test data";
        let key = [42u8; 32];
        let (enc1, k1) = encrypt_weights(data, Some(&key));
        let (enc2, k2) = encrypt_weights(data, Some(&key));
        assert_eq!(k1, key);
        assert_eq!(k2, key);
        assert_eq!(enc1, enc2);
    }

    #[test]
    fn commitment_deterministic() {
        let data = b"test data";
        assert_eq!(commitment(data), commitment(data));
        assert_ne!(commitment(b"a"), commitment(b"b"));
    }

    #[test]
    fn unit_conversion() {
        assert_eq!(to_shannons(1.0), SHANNONS_PER_SOMA);
        assert_eq!(to_soma(SHANNONS_PER_SOMA), 1.0);
        assert_eq!(to_shannons(0.5), SHANNONS_PER_SOMA / 2);
    }
}
