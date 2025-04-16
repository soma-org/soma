use super::{shard::Shard, shard_verifier::ShardAuthToken};
use crate::error::{ShardError, ShardResult};
use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::{keys::EncoderPublicKey, EncryptionKey},
    digest::Digest,
    error::SharedResult,
    metadata::{verify_metadata, EncryptionAPI, Metadata, MetadataAPI},
    scope::Scope,
    signed::Signed,
};

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
    fn encoder(&self) -> &EncoderPublicKey;
    fn committer(&self) -> &EncoderPublicKey;
    fn route(&self) -> &Option<Signed<Route, min_sig::BLS12381Signature>>;
    fn reveal_key_digest(&self) -> ShardResult<Digest<EncryptionKey>>;
    fn commit_metadata(&self) -> &Metadata;
}
impl ShardCommit {
    pub(crate) fn new_v1(
        auth_token: ShardAuthToken,
        encoder: EncoderPublicKey,
        route: Option<Signed<Route, min_sig::BLS12381Signature>>,
        commit: Metadata,
    ) -> ShardCommit {
        ShardCommit::V1(ShardCommitV1 {
            auth_token,
            encoder,
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
    encoder: EncoderPublicKey,
    // signed by the source (eligible inference encoder)
    route: Option<Signed<Route, min_sig::BLS12381Signature>>,
    commit: Metadata,
}

impl ShardCommitAPI for ShardCommitV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }

    fn committer(&self) -> &EncoderPublicKey {
        self.route
            .as_ref()
            .map(|r| r.destination())
            .unwrap_or(self.encoder())
    }
    fn route(&self) -> &Option<Signed<Route, min_sig::BLS12381Signature>> {
        &self.route
    }
    fn commit_metadata(&self) -> &Metadata {
        &self.commit
    }
    fn reveal_key_digest(&self) -> ShardResult<Digest<EncryptionKey>> {
        // TODO: remove encryption from metadata and add to the commit
        match self.commit.encryption() {
            Some(encryption) => Ok(encryption.key_digest()),
            None => Err(ShardError::EncryptionFailed),
        }
    }
}

pub(crate) fn verify_signed_shard_commit(
    signed_shard_commit: &Signed<ShardCommit, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    let auth_token_digest = Digest::new(signed_shard_commit.auth_token())?;
    // check that inference_encoder of the commit is inside the shards inference set
    if !shard.contains(&signed_shard_commit.encoder()) {
        return Err(shared::error::SharedError::ValidationError(
            "shard commit inference encoder is not in shard".to_string(),
        ));
    }
    // verify the commits metadata
    // TODO: verify the metadata is in-line with the requirement per the auth token
    verify_metadata(
        signed_shard_commit.commit_metadata(),
        None,
        None,
        None,
        None,
    )?;

    // If there exists a route, we need to verify it, otherwise skip
    if let Some(signed_route) = signed_shard_commit.route() {
        // if the route destination encoder already has a role in the shard, this should
        // cause an error
        if shard.contains(signed_route.destination()) {
            return Err(shared::error::SharedError::ValidationError(
                "route destination may not be a member of the shard".to_string(),
            ));
        }

        // the digest of the auth token supplied with the commit must match the digest included in the signed
        // route message. By forcing a signature of this digest, signed route messages cannot be replayed for different shards
        if signed_route.auth_token_digest() != auth_token_digest {
            return Err(shared::error::SharedError::ValidationError("s".to_string()));
        }

        // check signature of route is by the slot
        // the original slot must sign off on the route in order to be eligible
        let _ = signed_route.verify(
            Scope::ShardCommitRoute,
            signed_shard_commit.encoder().inner(),
        )?;
    }

    let _ =
        signed_shard_commit.verify(Scope::ShardCommit, signed_shard_commit.committer().inner())?;

    Ok(())
}
