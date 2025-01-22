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

    /// retrieves the signed shard input
    fn read_signed_shard_input(
        &self,
        shard_ref: &ShardRef,
    ) -> ShardResult<Verified<Signed<ShardInput>>>;

    /// retrieves the commit digest for a shard/peer pairing
    fn read_shard_commit_digest(
        &self,
        shard_ref: &ShardRef,
        peer: NetworkingIndex,
    ) -> ShardResult<Digest<Signed<ShardCommit>>>;

    /// batch retrieves the shard commit certificates for the list of peers
    fn batch_read_shard_commit_certificates(
        &self,
        shard_ref: ShardRef,
        peers: &[NetworkingIndex],
    ) -> ShardResult<Vec<Option<Verified<Certified<Signed<ShardCommit>>>>>>;

    /// retrieves the reveal digest for a shard/peer pair
    fn read_shard_reveal_digest(
        &self,
        shard_ref: &ShardRef,
        peer: NetworkingIndex,
    ) -> ShardResult<Digest<Signed<ShardReveal>>>;

    /// batch retrieves the shard reveal certificates for the list of peers
    fn batch_read_shard_reveal_certificates(
        &self,
        shard_ref: ShardRef,
        peers: &[NetworkingIndex],
    ) -> ShardResult<Vec<Option<Verified<Certified<Signed<ShardReveal>>>>>>;
}
