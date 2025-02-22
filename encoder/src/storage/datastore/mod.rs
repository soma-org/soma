#![doc = include_str!("README.md")]

pub(crate) mod mem_store;

use fastcrypto::bls12381::min_sig;
use shared::{crypto::EncryptionKey, digest::Digest, signed::Signed};

use crate::{
    error::ShardResult,
    types::{
        encoder_committee::{EncoderIndex, Epoch},
        shard::Shard,
        shard_commit::ShardCommit,
    },
};

/// The store is a common interface for accessing encoder data
pub(crate) trait Store: Send + Sync + 'static {
    fn atomic_commit(
        &self,
        shard_ref: Digest<Shard>,
        signed_commit: Signed<ShardCommit, min_sig::BLS12381Signature>,
    ) -> ShardResult<()>;
    fn check_reveal(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
        reveal_ref: Digest<EncryptionKey>,
    ) -> ShardResult<()>;
}
