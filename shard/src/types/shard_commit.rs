use crate::{
    crypto::{
        keys::{AuthorityPublicKey, ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
        DefaultHashFunction, DIGEST_LENGTH,
    },
    error::{ShardError, ShardResult},
};

use std::ops::Deref;
use std::sync::Arc;

use crate::types::manifest::Manifest;
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};

use super::{
    scope::{Scope, ScopedMessage},
    shard::ShardRef,
};

use std::{
    fmt,
    hash::{Hash, Hasher},
};

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardCommitAPI)]
pub enum ShardCommit {
    V1(ShardCommitV1),
}

/// `ShardCommitAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub(crate) trait ShardCommitAPI {
    /// returns the shard ref
    fn shard_ref(&self) -> &ShardRef;
    /// returns the manifest (checksums of the actual embeddings)
    fn manifest(&self) -> &Manifest;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardCommitV1 {
    /// manifest is the checksum / url references for the embeddings
    /// note the embeddings are not released for download yet so this is
    /// effectively just a commit hash with some additional metadata
    manifest: Manifest,
    /// shard ref, this is important for protecting against replay attacks
    shard_ref: ShardRef,
}

impl ShardCommitV1 {
    /// create a shard commit v1
    pub(crate) const fn new(shard_ref: ShardRef, manifest: Manifest) -> Self {
        Self {
            manifest,
            shard_ref,
        }
    }
}

impl ShardCommitAPI for ShardCommitV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
}

// pub struct CommitCertificate {
//     signed_commit: SignedShardCommit,
//     aggregate_signature: AuthorityPublicKey,
// }
