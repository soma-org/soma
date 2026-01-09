use crate::{
    metadata::{DownloadMetadata, ObjectPath},
    shard_crypto::keys::EncoderPublicKey,
};
use bytes::Bytes;
use enum_dispatch::enum_dispatch;
use fixed::types::U32F32;
use rand::Rng as _;
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
    fn target_embedding(&self) -> &Option<Embedding>;
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
    target_embedding: Option<Embedding>,
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
        target_embedding: Option<Embedding>,
    ) -> Self {
        Self {
            input_download_metadata,
            input_object_path,
            embedding_download_metadata,
            embedding_object_path,
            probe_encoder,
            probe_download_metadata,
            probe_object_path,
            target_embedding,
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
    fn target_embedding(&self) -> &Option<Embedding> {
        &self.target_embedding
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
    pub target_details: Option<TargetDetails>,
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

    pub fn mock() -> Self {
        let mut rng = rand::thread_rng();

        // Create the evaluation scores with random values between 0 and 1
        let evaluation_scores = EvaluationScores::V1(EvaluationScoresV1::new(
            U32F32::from_num(rng.gen::<f64>()), // flow_matching
            U32F32::from_num(rng.gen::<f64>()), // sig_reg
            U32F32::from_num(rng.gen::<f64>()), // compression
            U32F32::from_num(rng.gen::<f64>()), // composite
        ));

        // Create embeddings with random bytes
        let summary_embedding: Embedding = Bytes::from(rng.gen::<[u8; 32]>().to_vec());
        let sampled_embedding: Embedding = Bytes::from(rng.gen::<[u8; 32]>().to_vec());

        // Create target scores
        let target_scores = TargetScores::V1(TargetScoresV1::new(
            U32F32::from_num(rng.gen::<f64>()), // distance
            U32F32::from_num(rng.gen::<f64>()), // evaluation_score
            U32F32::from_num(rng.gen::<f64>()), // composite
        ));

        // Create target details
        let target_embedding: Embedding = Bytes::from(rng.gen::<[u8; 32]>().to_vec());
        let target_details =
            TargetDetails::V1(TargetDetailsV1::new(target_scores, target_embedding));

        EvaluationOutputV1::new(
            evaluation_scores,
            summary_embedding,
            sampled_embedding,
            Some(target_details),
        )
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

pub type FixedNum = fixed::types::U32F32;

#[enum_dispatch]
pub trait EvaluationScoresAPI {
    fn score(&self) -> FixedNum;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(EvaluationScoresAPI)]
pub enum EvaluationScores {
    V1(EvaluationScoresV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd, Eq)]
pub struct EvaluationScoresV1 {
    flow_matching: FixedNum,
    sig_reg: FixedNum,
    compression: FixedNum,
    composite: FixedNum,
}
impl EvaluationScoresV1 {
    pub fn new(
        flow_matching: FixedNum,
        sig_reg: FixedNum,
        compression: FixedNum,
        composite: FixedNum,
    ) -> Self {
        Self {
            flow_matching,
            sig_reg,
            compression,
            composite,
        }
    }
}

impl EvaluationScoresAPI for EvaluationScoresV1 {
    fn score(&self) -> FixedNum {
        self.composite
    }
}

#[enum_dispatch]
pub trait TargetScoresAPI {
    fn score(&self) -> FixedNum;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[enum_dispatch(TargetScoresAPI)]
pub enum TargetScores {
    V1(TargetScoresV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd, Eq, Hash)]
pub struct TargetScoresV1 {
    distance: FixedNum,
    evaluation_score: FixedNum,
    composite: FixedNum,
}

impl TargetScoresV1 {
    pub fn new(distance: FixedNum, evaluation_score: FixedNum, composite: FixedNum) -> Self {
        Self {
            distance,
            evaluation_score,
            composite,
        }
    }
}

impl TargetScoresAPI for TargetScoresV1 {
    fn score(&self) -> FixedNum {
        self.composite
    }
}

#[enum_dispatch]
pub trait TargetDetailsAPI {
    fn target_scores(&self) -> &TargetScores;
    fn target_embedding(&self) -> &Embedding;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(TargetDetailsAPI)]
pub enum TargetDetails {
    V1(TargetDetailsV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct TargetDetailsV1 {
    target_scores: TargetScores,
    target_embedding: Embedding,
}
impl TargetDetailsV1 {
    pub fn new(target_scores: TargetScores, target_embedding: Embedding) -> Self {
        Self {
            target_scores,
            target_embedding,
        }
    }
}

impl TargetDetailsAPI for TargetDetailsV1 {
    fn target_scores(&self) -> &TargetScores {
        &self.target_scores
    }
    fn target_embedding(&self) -> &Embedding {
        &self.target_embedding
    }
}

pub type Embedding = Bytes;
