use crate::encoder_committee::EncoderCommittee;
use crate::error::SharedResult;
use crate::evaluation::{EmbeddingDigest, Score};
use crate::metadata::{verify_metadata, DownloadMetadata, Metadata};
use crate::{shard::Shard, shard_crypto::digest::Digest, shard_crypto::keys::EncoderPublicKey};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait SubmissionAPI {
    fn encoder(&self) -> &EncoderPublicKey;
    fn shard_digest(&self) -> Digest<Shard>;
    fn metadata(&self) -> &Metadata;
    fn probe_encoder(&self) -> &EncoderPublicKey;
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
    input_download_metadata: DownloadMetadata,
    embedding_download_metadata: DownloadMetadata,
    probe_encoder: EncoderPublicKey,
    evaluation_scores: EvaluationScores,
    target_details: Option<TargetDetails>,
    summary_embedding: Embedding,
    sampled_embedding: Embedding,
}

impl SubmissionV1 {
    pub fn new(
        encoder: EncoderPublicKey,
        shard_digest: Digest<Shard>,
        metadata: Metadata,
        probe_encoder: EncoderPublicKey,
        score: Score,
        summary_digest: EmbeddingDigest,
    ) -> Self {
        Self {
            encoder,
            shard_digest,
            metadata,
            probe_encoder,
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
    fn probe_encoder(&self) -> &EncoderPublicKey {
        &self.probe_encoder
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

    // TODO: optionally add constraint on probe_encoder's stake being greater than or equal to the encoder
    Ok(())
}
