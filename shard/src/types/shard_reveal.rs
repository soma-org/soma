use crate::crypto::AesKey;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::shard::ShardRef;

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardRevealAPI)]
pub enum ShardReveal {
    V1(ShardRevealV1),
}

/// `ShardRevealAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
trait ShardRevealAPI {
    /// returns the shard ref
    fn shard_ref(&self) -> &ShardRef;
    /// returns the encryption key
    fn key(&self) -> &AesKey;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardRevealV1 {
    shard_ref: ShardRef,
    key: AesKey,
}

impl ShardRevealV1 {
    pub(crate) const fn new(shard_ref: ShardRef, key: AesKey) -> Self {
        Self { shard_ref, key }
    }
}

impl ShardRevealAPI for ShardRevealV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn key(&self) -> &AesKey {
        &self.key
    }
}
