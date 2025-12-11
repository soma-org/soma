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
    fn evaluation_scores(&self) -> &EvaluationScores;
    fn summary_embedding(&self) -> &Embedding;
    fn sampled_embedding(&self) -> &Embedding;
    fn target_details(&self) -> &Option<TargetDetails>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[enum_dispatch(EvaluationOutputAPI)]
pub enum EvaluationOutput {
    V1(EvaluationOutputV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct EvaluationOutputV1 {
    evaluation_scores: EvaluationScores,
    summary_embedding: Embedding,
    sampled_embedding: Embedding,
    target_details: Option<TargetDetails>,
}

impl EvaluationOutputV1 {
    pub fn new(
        evaluation_scores: EvaluationScores,
        summary_embedding: Embedding,
        sampled_embedding: Embedding,
        target_details: Option<TargetDetails>,
    ) -> Self {
        Self {
            evaluation_scores,
            summary_embedding,
            sampled_embedding,
            target_details,
        }
    }
}

impl EvaluationOutputAPI for EvaluationOutputV1 {
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

#[enum_dispatch]
pub trait EvaluationScoresAPI {
    fn value(&self) -> u64;
}

// TODO: convert this to use fixed point math directly!
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(EvaluationScoresAPI)]
pub enum EvaluationScores {
    V1(EvaluationScoresV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd, Eq)]
pub struct EvaluationScoresV1 {
    flow_matching: u64,
    sig_reg: u64,
    compression: u64,
    composite: u64,
}
impl EvaluationScoresV1 {
    pub fn new(flow_matching: u64, sig_reg: u64, compression: u64, composite: u64) -> Self {
        Self {
            flow_matching,
            sig_reg,
            compression,
            composite,
        }
    }
}

impl EvaluationScoresAPI for EvaluationScoresV1 {
    fn value(&self) -> u64 {
        self.composite
    }
}

#[enum_dispatch]
pub trait TargetScoresAPI {
    fn value(&self) -> u64;
}

// TODO: convert this to use fixed point math directly!
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(TargetScoresAPI)]
pub enum TargetScores {
    V1(TargetScoresV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd, Eq)]
pub struct TargetScoresV1 {
    value: u64,
}
impl TargetScoresV1 {
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}

impl TargetScoresAPI for TargetScoresV1 {
    fn value(&self) -> u64 {
        self.value
    }
}

// TODO: change this to actually be accurate
pub type Embedding = Digest<Vec<u8>>;

#[enum_dispatch]
pub trait TargetDetailsAPI {
    fn value(&self) -> u64;
}

// TODO: convert this to use fixed point math directly!
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(TargetDetailsAPI)]
pub enum TargetDetails {
    V1(TargetDetailsV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd, Eq)]
pub struct TargetDetailsV1 {
    value: u64,
}
impl TargetDetailsV1 {
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}

impl TargetDetailsAPI for TargetDetailsV1 {
    fn value(&self) -> u64 {
        self.value
    }
}
