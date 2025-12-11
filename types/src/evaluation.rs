use crate::{
    metadata::{DownloadMetadata, ObjectPath},
    shard_crypto::{digest::Digest, keys::EncoderPublicKey},
};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait EvaluationInputAPI {
    fn input_download_metadata(&self) -> &DownloadMetadata;
    fn input_object_path(&self) -> &ObjectPath;
    fn embedding_download_metadata(&self) -> &DownloadMetadata;
    fn embedding_object_path(&self) -> &ObjectPath;
    fn probe_encoder(&self) -> &EncoderPublicKey;
    fn probe_download_metadata(&self) -> &DownloadMetadata;
    fn probe_object_path(&self) -> &ObjectPath;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(EvaluationInputAPI)]
pub enum EvaluationInput {
    V1(EvaluationInputV1),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvaluationInputV1 {
    input_download_metadata: DownloadMetadata,
    input_object_path: ObjectPath,
    embedding_download_metadata: DownloadMetadata,
    embedding_object_path: ObjectPath,
    probe_encoder: EncoderPublicKey,
    probe_download_metadata: DownloadMetadata,
    probe_object_path: ObjectPath,
}

impl EvaluationInputV1 {
    pub fn new(
        input_download_metadata: DownloadMetadata,
        input_object_path: ObjectPath,
        embedding_download_metadata: DownloadMetadata,
        embedding_object_path: ObjectPath,
        probe_encoder: EncoderPublicKey,
        probe_download_metadata: DownloadMetadata,
        probe_object_path: ObjectPath,
    ) -> Self {
        Self {
            input_download_metadata,
            input_object_path,
            embedding_download_metadata,
            embedding_object_path,
            probe_encoder,
            probe_download_metadata,
            probe_object_path,
        }
    }
}

impl EvaluationInputAPI for EvaluationInputV1 {
    fn input_download_metadata(&self) -> &DownloadMetadata {
        &self.input_download_metadata
    }

    fn input_object_path(&self) -> &ObjectPath {
        &self.input_object_path
    }

    fn embedding_download_metadata(&self) -> &DownloadMetadata {
        &self.embedding_download_metadata
    }

    fn embedding_object_path(&self) -> &ObjectPath {
        &self.embedding_object_path
    }

    fn probe_encoder(&self) -> &EncoderPublicKey {
        &self.probe_encoder
    }
    fn probe_download_metadata(&self) -> &DownloadMetadata {
        &self.probe_download_metadata
    }
    fn probe_object_path(&self) -> &ObjectPath {
        &self.probe_object_path
    }
}

#[enum_dispatch]
pub trait EvaluationOutputAPI {
    fn score(&self) -> Score;
    fn summary_digest(&self) -> &EmbeddingDigest;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[enum_dispatch(EvaluationOutputAPI)]
pub enum EvaluationOutput {
    V1(EvaluationOutputV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct EvaluationOutputV1 {
    score: Score,
    summary_digest: EmbeddingDigest,
}

impl EvaluationOutputV1 {
    pub fn new(score: Score, summary_digest: EmbeddingDigest) -> Self {
        Self {
            score,
            summary_digest,
        }
    }
}

impl EvaluationOutputAPI for EvaluationOutputV1 {
    fn score(&self) -> Score {
        self.score.clone()
    }
    fn summary_digest(&self) -> &EmbeddingDigest {
        &self.summary_digest
    }
}

#[enum_dispatch]
pub trait ScoreAPI {
    fn value(&self) -> u64;
}

// TODO: convert this to use fixed point math directly!
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ScoreAPI)]
pub enum Score {
    V1(ScoreV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd, Eq)]
pub struct ScoreV1 {
    value: u64,
}
impl ScoreV1 {
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}

impl ScoreAPI for ScoreV1 {
    fn value(&self) -> u64 {
        self.value
    }
}

// TODO: change this to actually be accurate
pub type EmbeddingDigest = Digest<Vec<u8>>;
