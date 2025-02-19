use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{metadata::Metadata, signed::Signed};

use super::{
    encoder_committee::EncoderIndex,
    shard_verifier::{Route, ShardAuthToken},
};

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
    fn auth_token(&self) -> &ShardAuthToken;
    fn slot(&self) -> EncoderIndex;
    fn committer(&self) -> EncoderIndex;
    fn route(&self) -> &Option<Signed<Route, min_sig::BLS12381Signature>>;
    fn commit(&self) -> &Metadata;
}

impl ShardCommit {
    pub(crate) fn new_v1(
        auth_token: ShardAuthToken,
        slot: EncoderIndex,
        route: Option<Signed<Route, min_sig::BLS12381Signature>>,
        commit: Metadata,
    ) -> ShardCommit {
        ShardCommit::V1(ShardCommitV1 {
            auth_token,
            slot,
            route,
            commit,
        })
    }
}

//Digest<Signed<ShardCommit>>

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardCommitV1 {
    // the auth token protects against replay attacks since this entire thing is signed with
    // a unique shard auth token that is specific to the shard
    auth_token: ShardAuthToken,
    slot: EncoderIndex,
    // signed by the source (eligible inference encoder)
    route: Option<Signed<Route, min_sig::BLS12381Signature>>,
    commit: Metadata,
}

impl ShardCommitAPI for ShardCommitV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn slot(&self) -> EncoderIndex {
        self.slot
    }

    fn committer(&self) -> EncoderIndex {
        self.route
            .as_ref()
            .map(|r| r.destination())
            .unwrap_or(self.slot)
    }
    fn route(&self) -> &Option<Signed<Route, min_sig::BLS12381Signature>> {
        &self.route
    }
    fn commit(&self) -> &Metadata {
        &self.commit
    }
}
