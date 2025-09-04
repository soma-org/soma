use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use types::shard::Shard;
use types::shard::ShardAuthToken;
use types::{
    error::{SharedError, SharedResult},
    shard_crypto::{digest::Digest, keys::EncoderPublicKey, signed::Signed},
};

use super::reveal::Reveal;

#[enum_dispatch]
pub(crate) trait CommitAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn reveal_digest(&self) -> &Digest<Reveal>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct CommitV1 {
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    reveal_digest: Digest<Reveal>,
}

impl CommitV1 {
    pub(crate) fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        reveal_digest: Digest<Reveal>,
    ) -> Self {
        Self {
            auth_token,
            author,
            reveal_digest,
        }
    }
}

impl CommitAPI for CommitV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn author(&self) -> &EncoderPublicKey {
        &self.author
    }
    fn reveal_digest(&self) -> &Digest<Reveal> {
        &self.reveal_digest
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(CommitAPI)]
pub(crate) enum Commit {
    V1(CommitV1),
}

pub(crate) fn verify_commit(
    signed_commit: &Commit,
    peer: &EncoderPublicKey,
    shard: &Shard,
) -> SharedResult<()> {
    if peer != signed_commit.author() {
        return Err(SharedError::FailedTypeVerification(
            "sending peer must be author".to_string(),
        ));
    }

    // signed_commit.verify_signature(Scope::Commit, signed_commit.author().inner())?;
    Ok(())
}
