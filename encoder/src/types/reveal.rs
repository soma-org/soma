use std::collections::HashMap;

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use types::encoder_committee::EncoderCommittee;
use types::metadata::DownloadMetadata;
use types::submission::{verify_submission, Submission, SubmissionAPI};
use types::{
    error::{SharedError, SharedResult},
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
    fn embedding_download_metadata(&self) -> &DownloadMetadata;
    fn probe_set_download_metadata(&self) -> &HashMap<EncoderPublicKey, DownloadMetadata>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RevealV1 {
    auth_token: ShardAuthToken,
    author: EncoderPublicKey,
    submission: Submission,
    embedding_download_metadata: DownloadMetadata,
    probe_set_download_metadata: HashMap<EncoderPublicKey, DownloadMetadata>,
}

impl RevealV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        author: EncoderPublicKey,
        submission: Submission,
        embedding_download_metadata: DownloadMetadata,
        probe_set_download_metadata: HashMap<EncoderPublicKey, DownloadMetadata>,
    ) -> Self {
        Self {
            auth_token,
            author,
            submission,
            embedding_download_metadata,
            probe_set_download_metadata,
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
    fn embedding_download_metadata(&self) -> &DownloadMetadata {
        &self.embedding_download_metadata
    }
    fn probe_set_download_metadata(&self) -> &HashMap<EncoderPublicKey, DownloadMetadata> {
        &self.probe_set_download_metadata
    }
}

pub(crate) fn verify_reveal(
    reveal: &Reveal,
    peer: &EncoderPublicKey,
    shard: &Shard,
    encoder_committee: &EncoderCommittee,
) -> SharedResult<()> {
    if peer != reveal.author() {
        return Err(SharedError::FailedTypeVerification(
            "sending peer must be author".to_string(),
        ));
    }
    if peer != reveal.submission().encoder() {
        return Err(SharedError::FailedTypeVerification(
            "sending peer must be submission encoder".to_string(),
        ));
    }

    let _ = verify_submission(reveal.submission(), shard, encoder_committee)?;

    Ok(())
}
