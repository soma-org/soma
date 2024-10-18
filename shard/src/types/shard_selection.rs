use crate::{
    crypto::{
        keys::{ProtocolKeyPair, ProtocolKeySignature, ProtocolPublicKey},
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
    scope::{Scope, ScopedMessage},
    shard_commit::SignedShardCommit,
};

use std::{
    fmt,
    hash::{Hash, Hasher},
};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardSelectionAPI)]
pub enum ShardSelection {
    V1(ShardSelectionV1),
}

#[enum_dispatch]
pub trait ShardSelectionAPI {
    fn commits(&self) -> &[SignedShardCommit];
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardSelectionV1 {
    commits: Vec<SignedShardCommit>,
}

impl ShardSelectionV1 {
    pub(crate) fn new(commits: Vec<SignedShardCommit>) -> Self {
        Self { commits }
    }
}

impl ShardSelectionAPI for ShardSelectionV1 {
    fn commits(&self) -> &[SignedShardCommit] {
        &self.commits
    }
}
