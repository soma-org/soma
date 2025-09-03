pub mod messaging;
pub mod modules;
pub mod parameters;

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::keys::EncoderPublicKey,
    metadata::{DownloadableMetadata, Metadata},
};

#[enum_dispatch]
pub(crate) trait EvaluationInputAPI {
    fn data(&self) -> Metadata;
    fn embeddings(&self) -> Metadata;
    fn probe_set(&self) -> ProbeSet;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(EvaluationInputAPI)]
pub enum EvaluationInput {
    V1(EvaluationInputV1),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvaluationInputV1 {
    data: Metadata,
    embeddings: Metadata,
    probe_set: ProbeSet,
}

impl EvaluationInputV1 {
    pub fn new(data: Metadata, embeddings: Metadata, probe_set: ProbeSet) -> Self {
        Self {
            data,
            embeddings,
            probe_set,
        }
    }
}

impl EvaluationInputAPI for EvaluationInputV1 {
    fn data(&self) -> Metadata {
        self.data.clone()
    }
    fn embeddings(&self) -> Metadata {
        self.embeddings.clone()
    }
    fn probe_set(&self) -> ProbeSet {
        self.probe_set.clone()
    }
}

#[enum_dispatch]
pub trait EvaluationOutputAPI {
    fn score(&self) -> EvaluationScore;
    fn summary_embedding(&self) -> SummaryEmbedding;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[enum_dispatch(EvaluationOutputAPI)]
pub enum EvaluationOutput {
    V1(EvaluationOutputV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct EvaluationOutputV1 {
    score: EvaluationScoreV1,
    summary_embedding: SummaryEmbeddingV1,
}

impl EvaluationOutputV1 {
    pub fn new(score: EvaluationScoreV1, summary_embedding: SummaryEmbeddingV1) -> Self {
        Self {
            score,
            summary_embedding,
        }
    }
}

impl EvaluationOutputAPI for EvaluationOutputV1 {
    fn score(&self) -> EvaluationScore {
        EvaluationScore::V1(self.score.clone())
    }
    fn summary_embedding(&self) -> SummaryEmbedding {
        SummaryEmbedding::V1(self.summary_embedding.clone())
    }
}

#[enum_dispatch]
pub trait EvaluationScoreAPI {
    fn value(&self) -> u64;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[enum_dispatch(EvaluationScoreAPI)]
pub enum EvaluationScore {
    V1(EvaluationScoreV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct EvaluationScoreV1 {
    value: u64,
}
impl EvaluationScoreV1 {
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}

impl EvaluationScoreAPI for EvaluationScoreV1 {
    fn value(&self) -> u64 {
        self.value
    }
}

#[enum_dispatch]
pub trait SummaryEmbeddingAPI {
    fn value(&self) -> Vec<u64>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[enum_dispatch(SummaryEmbeddingAPI)]
pub enum SummaryEmbedding {
    V1(SummaryEmbeddingV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct SummaryEmbeddingV1 {
    value: Vec<u64>,
}
impl SummaryEmbeddingV1 {
    pub fn new(value: Vec<u64>) -> Self {
        Self { value }
    }
}

impl SummaryEmbeddingAPI for SummaryEmbeddingV1 {
    fn value(&self) -> Vec<u64> {
        self.value.clone()
    }
}
#[enum_dispatch]
pub trait ProbeWeightAPI {
    fn encoder(&self) -> &EncoderPublicKey;
    fn weight(&self) -> u64;
    fn downloadable_metadata(&self) -> DownloadableMetadata;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProbeWeightV1 {
    encoder: EncoderPublicKey,
    weight: u64,
    downloadable_metadata: DownloadableMetadata,
}

impl ProbeWeightAPI for ProbeWeightV1 {
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }

    fn weight(&self) -> u64 {
        self.weight
    }

    fn downloadable_metadata(&self) -> DownloadableMetadata {
        self.downloadable_metadata.clone()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ProbeWeightAPI)]
pub enum ProbeWeight {
    V1(ProbeWeightV1),
}

#[enum_dispatch]
pub trait ProbeSetAPI {
    fn probe_weights(&self) -> Vec<ProbeWeight>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ProbeSetAPI)]
pub enum ProbeSet {
    V1(ProbeSetV1),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProbeSetV1 {
    probe_weights: Vec<ProbeWeightV1>,
}
impl ProbeSetV1 {
    pub fn new(probe_weights: Vec<ProbeWeightV1>) -> Self {
        Self { probe_weights }
    }
}

impl ProbeSetAPI for ProbeSetV1 {
    fn probe_weights(&self) -> Vec<ProbeWeight> {
        self.probe_weights
            .iter()
            .map(|pw| ProbeWeight::V1(pw.clone()))
            .collect()
    }
}
