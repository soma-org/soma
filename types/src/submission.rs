use crate::encoder_committee::EncoderCommittee;
use crate::error::SharedResult;
use crate::evaluation::{Embedding, EvaluationScores, TargetDetails};
use crate::metadata::{verify_metadata, DownloadMetadata, Metadata};
use crate::{shard::Shard, shard_crypto::digest::Digest, shard_crypto::keys::EncoderPublicKey};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait SubmissionAPI {
    fn encoder(&self) -> &EncoderPublicKey;
    fn shard_digest(&self) -> &Digest<Shard>;
    fn input_download_metadata(&self) -> &DownloadMetadata;
    fn embedding_download_metadata(&self) -> &DownloadMetadata;
    fn probe_encoder(&self) -> &EncoderPublicKey;
    fn evaluation_scores(&self) -> &EvaluationScores;
    fn summary_embedding(&self) -> &Embedding;
    fn sampled_embedding(&self) -> &Embedding;
    fn target_details(&self) -> &Option<TargetDetails>;
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
    summary_embedding: Embedding,
    sampled_embedding: Embedding,
    target_details: Option<TargetDetails>,
}

impl SubmissionV1 {
    pub fn new(
        encoder: EncoderPublicKey,
        shard_digest: Digest<Shard>,
        input_download_metadata: DownloadMetadata,
        embedding_download_metadata: DownloadMetadata,
        probe_encoder: EncoderPublicKey,
        evaluation_scores: EvaluationScores,
        summary_embedding: Embedding,
        sampled_embedding: Embedding,
        target_details: Option<TargetDetails>,
    ) -> Self {
        Self {
            encoder,
            shard_digest,
            input_download_metadata,
            embedding_download_metadata,
            probe_encoder,
            evaluation_scores,
            summary_embedding,
            sampled_embedding,
            target_details,
        }
    }
}

impl SubmissionAPI for SubmissionV1 {
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }
    fn shard_digest(&self) -> &Digest<Shard> {
        &self.shard_digest
    }
    fn input_download_metadata(&self) -> &DownloadMetadata {
        &self.input_download_metadata
    }
    fn embedding_download_metadata(&self) -> &DownloadMetadata {
        &self.embedding_download_metadata
    }
    fn probe_encoder(&self) -> &EncoderPublicKey {
        &self.probe_encoder
    }
    fn evaluation_scores(&self) -> &EvaluationScores {
        &self.evaluation_scores
    }
    fn summary_embedding(&self) -> &Embedding {
        &self.summary_embedding
    }
    fn sampled_embedding(&self) -> &Embedding {
        &self.sampled_embedding
    }
    fn target_details(&self) -> &Option<TargetDetails> {
        &self.target_details
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
    if shard.digest()? != *submission.shard_digest() {
        // return error
    }

    // TODO: optionally add constraint on probe_encoder's stake being greater than or equal to the encoder
    Ok(())
}
