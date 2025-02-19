#![doc = include_str!("README.md")]

pub(crate) mod mem_store;

use fastcrypto::bls12381::min_sig;
use shared::{
    digest::Digest, network_committee::NetworkingIndex, signed::Signed, verified::Verified,
};

use crate::{
    error::ShardResult,
    types::{
        certified::Certified,
        shard::{Shard, ShardRef},
        shard_commit::ShardCommit,
        shard_input::ShardInput,
        shard_reveal::ShardReveal,
    },
};

/// The store is a common interface for accessing encoder data
pub(crate) trait Store: Send + Sync + 'static {
    fn atomic_commit(
        &self,
        shard_ref: Digest<Shard>,
        signed_commit: Signed<ShardCommit, min_sig::BLS12381Signature>,
    ) -> ShardResult<()>;
}
