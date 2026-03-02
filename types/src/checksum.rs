// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::hash::{Hash, Hasher};

use fastcrypto::error::FastCryptoError;
use fastcrypto::traits::ToFromBytes;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::crypto::DIGEST_LENGTH;
use crate::digests::Digest;

/// Checksum is a bytes checksum for data. We use the same default hash function
/// as the rest of the network. There are associated functions for new from bytes
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct Checksum(Digest);

impl Checksum {
    pub(crate) const MIN: Self = Self(Digest::new([u8::MIN; DIGEST_LENGTH]));
    pub(crate) const MAX: Self = Self(Digest::new([u8::MAX; DIGEST_LENGTH]));

    pub fn new_from_hash(hash: [u8; DIGEST_LENGTH]) -> Self {
        Self(Digest::new(hash))
    }
}

impl Hash for Checksum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.inner()[..8]);
    }
}

impl From<Checksum> for fastcrypto::hash::Digest<{ DIGEST_LENGTH }> {
    fn from(hd: Checksum) -> Self {
        fastcrypto::hash::Digest::new(hd.0.into_inner())
    }
}

impl From<Checksum> for [u8; 32] {
    fn from(checksum: Checksum) -> Self {
        checksum.0.into_inner()
    }
}

impl fmt::Display for Checksum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::Debug for Checksum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(&self.0, f)
    }
}

impl AsRef<[u8]> for Checksum {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl ToFromBytes for Checksum {
    fn from_bytes(bytes: &[u8]) -> Result<Self, FastCryptoError> {
        if bytes.len() != DIGEST_LENGTH {
            return Err(FastCryptoError::InvalidInput);
        }
        let mut arr = [0u8; DIGEST_LENGTH];
        arr.copy_from_slice(bytes);
        Ok(Self(Digest::new(arr)))
    }

    fn as_bytes(&self) -> &[u8] {
        self.0.as_ref()
    }
}
