#![doc = include_str!("README.md")]

pub(crate) mod mem_store;

use std::time::Duration;

use fastcrypto::bls12381::min_sig;
use shared::{checksum::Checksum, crypto::EncryptionKey, digest::Digest, signed::Signed};

use crate::{
    error::ShardResult,
    types::{
        certified::Certified,
        encoder_committee::{EncoderIndex, Epoch},
        shard::Shard,
        shard_commit::ShardCommit,
    },
};

/// The store is a common interface for accessing encoder data
pub(crate) trait Store: Send + Sync + 'static {
    fn lock_signed_commit_digest(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
        committer: EncoderIndex,
        digest: Digest<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<()>;
    fn atomic_certified_commit(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
        certified_commit: Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>,
    ) -> ShardResult<usize>;
    fn get_certified_commit(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
    ) -> ShardResult<Certified<Signed<ShardCommit, min_sig::BLS12381Signature>>>;
    fn time_since_first_certified_commit(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
    ) -> Option<Duration>;
    fn atomic_reveal(
        &self,
        epoch: Epoch,
        shard_ref: Digest<Shard>,
        slot: EncoderIndex,
        reveal: EncryptionKey,
        checksum: Checksum,
    ) -> ShardResult<usize>;
    fn time_since_first_reveal(&self, epoch: Epoch, shard_ref: Digest<Shard>) -> Option<Duration>;
}
