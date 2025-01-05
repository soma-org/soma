use crate::{
    crypto::{
        keys::{AuthorityPublicKey, ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
        DefaultHashFunction, DIGEST_LENGTH,
    },
    error::{ShardError, ShardResult},
};

use std::ops::Deref;
use std::sync::Arc;

use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fastcrypto::hash::{Digest, HashFunction};
use serde::{Deserialize, Serialize};

use super::{
    data::Data,
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
    fn data(&self) -> &Data;
}

impl ShardCommit {
    pub(crate) fn new_v1(shard_ref: ShardRef, data: Data) -> ShardCommit {
        ShardCommit::V1(ShardCommitV1 { shard_ref, data })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardCommitV1 {
    data: Data,
    /// shard ref, this is important for protecting against replay attacks
    shard_ref: ShardRef,
}

impl ShardCommitAPI for ShardCommitV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn data(&self) -> &Data {
        &self.data
    }
}

// pub struct CommitCertificate {
//     signed_commit: SignedShardCommit,
//     aggregate_signature: AuthorityPublicKey,
// }
