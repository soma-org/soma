pub(crate) mod address;
pub(crate) mod keys;

use crypto_common::generic_array::GenericArray;
use fastcrypto::hash::{Blake2b256, HashFunction};

pub(crate) type DefaultHashFunction = Blake2b256;
pub(crate) const DIGEST_LENGTH: usize = DefaultHashFunction::OUTPUT_SIZE;

pub(crate) type AesKey = GenericArray<u8, <aes::Aes256 as crypto_common::KeySizeUser>::KeySize>;
