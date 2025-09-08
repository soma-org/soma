use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use types::shard::Shard;
use types::shard::ShardAuthToken;
use types::submission::Submission;
use types::{
    error::{SharedError, SharedResult},
    shard_crypto::{digest::Digest, keys::EncoderPublicKey},
};

#[enum_dispatch]
pub(crate) trait CommitAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn submission_digest(&self) -> &Digest<Submission>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct CommitV1 {
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    submission_digest: Digest<Submission>,
}

impl CommitV1 {
    pub(crate) fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        submission_digest: Digest<Submission>,
    ) -> Self {
        Self {
            auth_token,
            author,
            submission_digest,
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
    fn submission_digest(&self) -> &Digest<Submission> {
        &self.submission_digest
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
    Ok(())
}
