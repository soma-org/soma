#![doc = include_str!("README.md")]

pub(crate) mod mem_store;

use shared::{
    digest::Digest, network_committee::NetworkingIndex, signed::Signed, verified::Verified,
};

use crate::{
    error::ShardResult,
    types::{
        certified::Certified, shard::ShardRef, shard_commit::ShardCommit, shard_input::ShardInput,
        shard_reveal::ShardReveal,
    },
};

/// The store is a common interface for accessing encoder data
pub(crate) trait Store: Send + Sync + 'static {
    /// used to check whether the encoder has any knowledge of the shard
    fn contains_shard(&self, shard_ref: &ShardRef) -> ShardResult<()>;

    fn read_shard(&self, shard_ref: &ShardRef) -> ShardResult<Vec<NetworkingIndex>>;
}
