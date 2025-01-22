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
    shard_inputs: BTreeMap<ShardRef, Verified<Signed<ShardInput>>>,
    shard_commit_digests: BTreeMap<(ShardRef, NetworkingIndex), Digest<Signed<ShardCommit>>>,
    shard_commit_certificates:
        BTreeMap<(ShardRef, NetworkingIndex), Verified<Certified<Signed<ShardCommit>>>>,
    shard_reveal_digests: BTreeMap<(ShardRef, NetworkingIndex), Digest<Signed<ShardReveal>>>,
    shard_reveal_certificates:
        BTreeMap<(ShardRef, NetworkingIndex), Verified<Certified<Signed<ShardReveal>>>>,
}

impl MemStore {
    // #[cfg(test)]
    pub(crate) const fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                shards: BTreeMap::new(),
                shard_inputs: BTreeMap::new(),
                shard_commit_digests: BTreeMap::new(),
                shard_commit_certificates: BTreeMap::new(),
                shard_reveal_digests: BTreeMap::new(),
                shard_reveal_certificates: BTreeMap::new(),
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

    /// retrieves the signed shard input
    fn read_signed_shard_input(
        &self,
        shard_ref: &ShardRef,
    ) -> ShardResult<Verified<Signed<ShardInput>>> {
        let inner = self.inner.read();
        inner
            .shard_inputs
            .get(&shard_ref)
            .cloned()
            .ok_or(ShardError::DatastoreError(
                "shard input not found".to_string(),
            ))
    }

    /// retrieves the commit digest for a shard/peer pairing
    fn read_shard_commit_digest(
        &self,
        shard_ref: &ShardRef,
        peer: NetworkingIndex,
    ) -> ShardResult<Digest<Signed<ShardCommit>>> {
        let inner = self.inner.read();
        inner
            .shard_commit_digests
            .get(&(shard_ref.clone(), peer))
            .cloned()
            .ok_or(ShardError::DatastoreError(
                "shard commit digest not found".to_string(),
            ))
    }

    /// batch retrieves the shard commit certificates for the list of peers
    fn batch_read_shard_commit_certificates(
        &self,
        shard_ref: ShardRef,
        peers: &[NetworkingIndex],
    ) -> ShardResult<Vec<Option<Verified<Certified<Signed<ShardCommit>>>>>> {
        let inner = self.inner.read();
        let shard_commit_certificates = peers
            .iter()
            .map(|peer| {
                inner
                    .shard_commit_certificates
                    .get(&(shard_ref.clone(), *peer))
                    .cloned()
            })
            .collect();
        Ok(shard_commit_certificates)
    }

    /// retrieves the reveal digest for a shard/peer pair
    fn read_shard_reveal_digest(
        &self,
        shard_ref: &ShardRef,
        peer: NetworkingIndex,
    ) -> ShardResult<Digest<Signed<ShardReveal>>> {
        let inner = self.inner.read();
        inner
            .shard_reveal_digests
            .get(&(shard_ref.clone(), peer))
            .copied()
            .ok_or(ShardError::DatastoreError(
                "shard reveal digest not found".to_string(),
            ))
    }

    /// batch retrieves the shard reveal certificates for the list of peers
    fn batch_read_shard_reveal_certificates(
        &self,
        shard_ref: ShardRef,
        peers: &[NetworkingIndex],
    ) -> ShardResult<Vec<Option<Verified<Certified<Signed<ShardReveal>>>>>> {
        let inner = self.inner.read();
        let shard_commit_certificates = peers
            .iter()
            .map(|peer| {
                inner
                    .shard_reveal_certificates
                    .get(&(shard_ref.clone(), *peer))
                    .cloned()
            })
            .collect();
        Ok(shard_commit_certificates)
    }
}
