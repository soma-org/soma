use std::ops::Deref;

use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{crypto::keys::EncoderPublicKey, digest::Digest, metadata::Metadata, signed::Signed};

use super::{encoder_committee::InferenceEncoder, shard_verifier::ShardAuthToken};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Route {
    // the selected encoder to commit on their behalf
    destination: EncoderPublicKey,
    // digest is used to stop replay attacks
    auth_token_digest: Digest<ShardAuthToken>,
}

impl Route {
    pub fn destination(&self) -> &EncoderPublicKey {
        &self.destination
    }
    pub fn auth_token_digest(&self) -> Digest<ShardAuthToken> {
        self.auth_token_digest
    }
}

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ShardCommitAPI)]
pub enum ShardCommit {
    V1(ShardCommitV1),
}

/// `ShardCommitAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub(crate) trait ShardCommitAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn inference_encoder(&self) -> &InferenceEncoder;
    fn committer(&self) -> &EncoderPublicKey;
    fn route(&self) -> &Option<Signed<Route, min_sig::BLS12381Signature>>;
    // TODO: make this a wrapped version of metadata specific for embeddings!
    fn commit(&self) -> &Metadata;
}

impl ShardCommit {
    pub(crate) fn new_v1(
        auth_token: ShardAuthToken,
        inference_encoder: InferenceEncoder,
        route: Option<Signed<Route, min_sig::BLS12381Signature>>,
        commit: Metadata,
    ) -> ShardCommit {
        ShardCommit::V1(ShardCommitV1 {
            auth_token,
            inference_encoder,
            route,
            commit,
        })
    }
}

//Digest<Signed<ShardCommit>>

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ShardCommitV1 {
    // the auth token protects against replay attacks since this entire thing is signed with
    // a unique shard auth token that is specific to the shard
    auth_token: ShardAuthToken,
    inference_encoder: InferenceEncoder,
    // signed by the source (eligible inference encoder)
    route: Option<Signed<Route, min_sig::BLS12381Signature>>,
    commit: Metadata,
}

impl ShardCommitAPI for ShardCommitV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn inference_encoder(&self) -> &InferenceEncoder {
        &self.inference_encoder
    }

    fn committer(&self) -> &EncoderPublicKey {
        self.route
            .as_ref()
            .map(|r| r.destination())
            .unwrap_or(self.inference_encoder.deref())
    }
    fn route(&self) -> &Option<Signed<Route, min_sig::BLS12381Signature>> {
        &self.route
    }
    fn commit(&self) -> &Metadata {
        &self.commit
    }
}
