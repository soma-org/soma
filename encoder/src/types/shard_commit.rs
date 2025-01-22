use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::metadata::Metadata;

use super::{shard::ShardRef};

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
    fn data(&self) -> &Metadata;
}

impl ShardCommit {
    pub(crate) fn new_v1(shard_ref: ShardRef, data: Metadata) -> ShardCommit {
        ShardCommit::V1(ShardCommitV1 { shard_ref, data })
    }
}

//Digest<Signed<ShardCommit>>

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardCommitV1 {
    // transaction_proof
    // data (encrypted obviously)
    // the encryption option in the data will contain the hash of the encryption key
    // this means that you do not need to certify the reveal key, its baked in
    data: Metadata,
    /// shard ref, this is important for protecting against replay attacks
    shard_ref: ShardRef,
}

impl ShardCommitAPI for ShardCommitV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn data(&self) -> &Metadata {
        &self.data
    }
}

// pub struct CommitCertificate {
//     signed_commit: SignedShardCommit,
//     aggregate_signature: AuthorityPublicKey,
// }
