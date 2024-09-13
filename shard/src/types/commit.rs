use std::{
    fmt,
    hash::{Hash, Hasher},
};

use crate::crypto::DIGEST_LENGTH;
use fastcrypto::hash::Digest;
use serde::{Deserialize, Serialize};

/// CommitIndex references a specific commit in a vector
type CommitIndex = u32;

/// Represents the hash digest of a commit
#[derive(Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct CommitDigest([u8; DIGEST_LENGTH]);

impl CommitDigest {
    /// Lexicographic min digest.
    const MIN: Self = Self([u8::MIN; DIGEST_LENGTH]);
    /// Lexicographic max digest.
    const MAX: Self = Self([u8::MAX; DIGEST_LENGTH]);

    /// returns the inner digest as a byte array
    const fn into_inner(self) -> [u8; DIGEST_LENGTH] {
        self.0
    }
}

impl Hash for CommitDigest {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0[..8]);
    }
}

impl From<CommitDigest> for Digest<{ DIGEST_LENGTH }> {
    fn from(hd: CommitDigest) -> Self {
        Digest::new(hd.0)
    }
}

impl fmt::Display for CommitDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
                .get(0..4)
                .ok_or(fmt::Error)?
        )
    }
}

impl fmt::Debug for CommitDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            f,
            "{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, self.0)
        )
    }
}

/// Uniquely identifies a commit with its index and digest.
#[derive(Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct CommitRef {
    index: CommitIndex,
    digest: CommitDigest,
}

impl CommitRef {
    /// creates a new commit ref
    fn new(index: CommitIndex, digest: CommitDigest) -> Self {
        Self { index, digest }
    }
}

impl fmt::Display for CommitRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "C{}({})", self.index, self.digest)
    }
}

impl fmt::Debug for CommitRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "C{}({:?})", self.index, self.digest)
    }
}

/// Represents a vote on a Commit.
pub(crate) type CommitVote = CommitRef;
