use crate::{
    evaluation::{Embedding, EvaluationScores, TargetScores},
    metadata::DownloadMetadata,
    shard::Shard,
    shard_crypto::{digest::Digest, keys::EncoderPublicKey},
};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait ReportAPI {
    fn encoder(&self) -> &EncoderPublicKey;
    fn shard_digest(&self) -> &Digest<Shard>;
    fn input_download_metadata(&self) -> &DownloadMetadata;
    fn embedding_download_metadata(&self) -> &DownloadMetadata;
    fn probe_encoder(&self) -> &EncoderPublicKey;
    fn evaluation_scores(&self) -> &EvaluationScores;
    fn summary_embedding(&self) -> &Embedding;
    fn sampled_embedding(&self) -> &Embedding;
    fn target_scores(&self) -> &Option<TargetScores>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ReportAPI)]
pub enum Report {
    V1(ReportV1),
}

impl Report {
    pub fn as_v1(&self) -> &ReportV1 {
        match self {
            Self::V1(v1) => v1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ReportV1 {
    pub encoder: EncoderPublicKey,
    shard_digest: Digest<Shard>,
    pub input_download_metadata: DownloadMetadata,
    pub embedding_download_metadata: DownloadMetadata,
    probe_encoder: EncoderPublicKey,
    pub evaluation_scores: EvaluationScores,
    pub summary_embedding: Embedding,
    pub sampled_embedding: Embedding,
    pub target_scores: Option<TargetScores>,
}

impl ReportV1 {
    pub fn new(
        encoder: EncoderPublicKey,
        shard_digest: Digest<Shard>,
        input_download_metadata: DownloadMetadata,
        embedding_download_metadata: DownloadMetadata,
        probe_encoder: EncoderPublicKey,
        evaluation_scores: EvaluationScores,
        summary_embedding: Embedding,
        sampled_embedding: Embedding,
        target_scores: Option<TargetScores>,
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
            target_scores,
        }
    }
}

impl ReportAPI for ReportV1 {
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
    fn target_scores(&self) -> &Option<TargetScores> {
        &self.target_scores
    }
}
