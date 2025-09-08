use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use types::shard::Shard;
use types::shard::ShardAuthToken;
use types::{
    error::{SharedError, SharedResult},
    shard_crypto::{digest::Digest, keys::EncoderPublicKey},
    submission::Submission,
};

// reject votes are implicit
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(CommitVotesAPI)]
pub(crate) enum CommitVotes {
    V1(CommitVotesV1),
}

#[enum_dispatch]
pub(crate) trait CommitVotesAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn accepts(&self) -> &[(EncoderPublicKey, Digest<Submission>)];
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct CommitVotesV1 {
    /// stateless auth + stops replay attacks
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    accepts: Vec<(EncoderPublicKey, Digest<Submission>)>,
}

impl CommitVotesV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        accepts: Vec<(EncoderPublicKey, Digest<Submission>)>,
    ) -> Self {
        Self {
            auth_token,
            author,
            accepts,
        }
    }
}

impl CommitVotesAPI for CommitVotesV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn author(&self) -> &EncoderPublicKey {
        &self.author
    }
    fn accepts(&self) -> &[(EncoderPublicKey, Digest<Submission>)] {
        &self.accepts
    }
}

pub(crate) fn verify_commit_votes(
    commit_votes: &CommitVotes,
    peer: &EncoderPublicKey,
    shard: &Shard,
) -> SharedResult<()> {
    if peer != commit_votes.author() {
        return Err(SharedError::FailedTypeVerification(
            "sending peer must be author".to_string(),
        ));
    }

    let mut unique_encoders = HashSet::new();
    for (encoder, _commit_digest) in commit_votes.accepts() {
        if !unique_encoders.insert(encoder) {
            return Err(SharedError::ValidationError(format!(
                "redundant encoder detected: {:?}",
                encoder
            )));
        }
        if !shard.contains(encoder) {
            return Err(types::error::SharedError::ValidationError(
                "encoder not in shard".to_string(),
            ));
        }
    }

    Ok(())
}
