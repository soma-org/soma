use parking_lot::RwLock;
use shared::{
    digest::Digest, network_committee::NetworkingIndex, signed::Signed, verified::Verified,
};
use std::collections::BTreeMap;

use crate::{
    error::{ShardError, ShardResult},
    types::{
        certified::Certified, shard::ShardRef, shard_commit::ShardCommit, shard_input::ShardInput,
        shard_reveal::ShardReveal,
    },
};

use super::Store;

/// In-memory storage for testing.
#[allow(unused)]
pub(crate) struct MemStore {
    inner: RwLock<Inner>,
}

#[allow(unused)]
struct Inner {
    shards: BTreeMap<ShardRef, Vec<NetworkingIndex>>,
}

impl MemStore {
    // #[cfg(test)]
    pub(crate) const fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                shards: BTreeMap::new(),
            }),
        }
    }
}

impl Store for MemStore {
    /// used to check whether the encoder has any knowledge of the shard
    fn contains_shard(&self, shard_ref: &ShardRef) -> ShardResult<()> {
        let inner = self.inner.read();
        if inner.shards.contains_key(shard_ref) {
            return Ok(());
        }
        Err(ShardError::DatastoreError("shard not found".to_string()))
    }

    fn read_shard(&self, shard_ref: &ShardRef) -> ShardResult<Vec<NetworkingIndex>> {
        let inner = self.inner.read();
        inner
            .shards
            .get(shard_ref)
            .cloned()
            .ok_or(ShardError::DatastoreError("shard not found".to_string()))
    }
}
