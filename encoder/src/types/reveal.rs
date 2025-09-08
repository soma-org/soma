use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use types::submission::{Submission, SubmissionAPI};
use types::{
    error::{SharedError, SharedResult},
    metadata::verify_metadata,
    shard_crypto::keys::EncoderPublicKey,
};

use types::shard::Shard;
use types::shard::ShardAuthToken;

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(RevealAPI)]
pub enum Reveal {
    V1(RevealV1),
}

/// `RevealAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub(crate) trait RevealAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn author(&self) -> &EncoderPublicKey;
    fn submission(&self) -> &Submission;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RevealV1 {
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    submission: Submission,
}

impl RevealV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        submission: Submission,
    ) -> Self {
        Self {
            auth_token,
            author,
            submission,
        }
    }
}

impl RevealAPI for RevealV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn author(&self) -> &EncoderPublicKey {
        &self.author
    }
    fn submission(&self) -> &Submission {
        &self.submission
    }
}

pub(crate) fn verify_reveal(
    reveal: &Reveal,
    peer: &EncoderPublicKey,
    shard: &Shard,
) -> SharedResult<()> {
    if peer != reveal.author() {
        return Err(SharedError::FailedTypeVerification(
            "sending peer must be author".to_string(),
        ));
    }
    // Do I want to gurantee that the downloadable metadata match the peer and address on chain?

    // TODO: verify the probe_set's validity
    // TODO: verify the summary embedding's length

    let max_size = None;
    verify_metadata(&reveal.submission().metadata(), max_size)?;

    Ok(())
}
