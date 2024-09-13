pub(crate) mod address;
pub(crate) mod keys;

use fastcrypto::hash::{Blake2b256, HashFunction};

pub(crate) type DefaultHashFunction = Blake2b256;
pub(crate) const DIGEST_LENGTH: usize = DefaultHashFunction::OUTPUT_SIZE;
