use crate::encoder_committee::{self, EncoderCommittee};
use crate::error::SharedResult;
use crate::evaluation::{verify_probe_set, EmbeddingDigest, ProbeSet, Score};
use crate::metadata::{verify_metadata, Metadata};
use crate::{shard::Shard, shard_crypto::digest::Digest, shard_crypto::keys::EncoderPublicKey};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait SubmissionAPI {
    fn encoder(&self) -> &EncoderPublicKey;
    fn shard_digest(&self) -> Digest<Shard>;
    fn metadata(&self) -> &Metadata;
    fn probe_set(&self) -> &ProbeSet;
    fn score(&self) -> &Score;
    fn summary_digest(&self) -> &EmbeddingDigest;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(SubmissionAPI)]
pub enum Submission {
    V1(SubmissionV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SubmissionV1 {
    encoder: EncoderPublicKey,
    shard_digest: Digest<Shard>,
    metadata: Metadata,
    probe_set: ProbeSet,
    score: Score,
    summary_digest: EmbeddingDigest,
}

impl SubmissionV1 {
    pub fn new(
        encoder: EncoderPublicKey,
        shard_digest: Digest<Shard>,
        metadata: Metadata,
        probe_set: ProbeSet,
        score: Score,
        summary_digest: EmbeddingDigest,
    ) -> Self {
        Self {
            encoder,
            shard_digest,
            metadata,
            probe_set,
            score,
            summary_digest,
        }
    }
}

impl SubmissionAPI for SubmissionV1 {
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }
    fn score(&self) -> &Score {
        &self.score
    }
    fn summary_digest(&self) -> &EmbeddingDigest {
        &self.summary_digest
    }
    fn probe_set(&self) -> &ProbeSet {
        &self.probe_set
    }
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    fn shard_digest(&self) -> Digest<Shard> {
        self.shard_digest.clone()
    }
}

pub fn verify_submission(
    submission: &Submission,
    shard: &Shard,
    encoder_committee: &EncoderCommittee,
) -> SharedResult<()> {
    if !shard.contains(submission.encoder()) {
        // return error
    }
    if shard.digest()? != submission.shard_digest() {
        // return error
    }
    let _ = verify_metadata(submission.metadata(), None)?;
    let _ = verify_probe_set(submission.probe_set(), encoder_committee)?;

    Ok(())
}
