pub mod messaging;
pub mod modules;
pub mod parameters;
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::{
    crypto::{keys::EncoderPublicKey, EncryptionKey},
    metadata::Metadata,
};

#[enum_dispatch]
pub(crate) trait ProbeInputAPI {
    fn embeddings(&self) -> Vec<Embedding>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ProbeInputAPI)]
pub enum ProbeInput {
    V1(ProbeInputV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProbeInputV1 {
    embeddings: Vec<EmbeddingV1>,
}

impl ProbeInputV1 {
    pub fn new(embeddings: Vec<EmbeddingV1>) -> Self {
        Self { embeddings }
    }
}

impl ProbeInputAPI for ProbeInputV1 {
    fn embeddings(&self) -> Vec<Embedding> {
        self.embeddings
            .iter()
            .map(|e| Embedding::V1(e.clone()))
            .collect()
    }
}

#[enum_dispatch]
pub trait EmbeddingAPI {
    fn encoder(&self) -> &EncoderPublicKey;
    fn committer(&self) -> &EncoderPublicKey;
    fn probe(&self) -> &Metadata;
    fn commit(&self) -> &Metadata;
    fn reveal_key(&self) -> &EncryptionKey;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(EmbeddingAPI)]
pub enum Embedding {
    V1(EmbeddingV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct EmbeddingV1 {
    encoder: EncoderPublicKey,
    committer: EncoderPublicKey,
    probe: Metadata,
    commit: Metadata,
    reveal_key: EncryptionKey,
}

impl EmbeddingV1 {
    pub fn new(
        encoder: EncoderPublicKey,
        committer: EncoderPublicKey,
        probe: Metadata,
        commit: Metadata,
        reveal_key: EncryptionKey,
    ) -> Self {
        Self {
            encoder,
            committer,
            probe,
            commit,
            reveal_key,
        }
    }
}

impl EmbeddingAPI for EmbeddingV1 {
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }
    fn committer(&self) -> &EncoderPublicKey {
        &self.committer
    }
    fn probe(&self) -> &Metadata {
        &self.probe
    }
    fn commit(&self) -> &Metadata {
        &self.commit
    }
    fn reveal_key(&self) -> &EncryptionKey {
        &self.reveal_key
    }
}

#[enum_dispatch]
pub trait ProbeOutputAPI {
    fn scores(&self) -> Vec<Score>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ProbeOutputAPI)]
pub enum ProbeOutput {
    V1(ProbeOutputV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProbeOutputV1 {
    scores: Vec<ScoreV1>,
}

impl ProbeOutputV1 {
    pub fn new(mut scores: Vec<ScoreV1>) -> Self {
        scores.sort();
        Self { scores }
    }
}

impl ProbeOutputAPI for ProbeOutputV1 {
    fn scores(&self) -> Vec<Score> {
        self.scores.iter().map(|s| Score::V1(s.clone())).collect()
    }
}

#[enum_dispatch]
pub trait ScoreAPI {
    fn encoder(&self) -> &EncoderPublicKey;
    fn rank(&self) -> u8;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ScoreAPI)]
pub enum Score {
    V1(ScoreV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ScoreV1 {
    encoder: EncoderPublicKey,
    rank: u8,
}
impl ScoreV1 {
    pub fn new(encoder: EncoderPublicKey, rank: u8) -> Self {
        Self { encoder, rank }
    }
}

impl ScoreAPI for ScoreV1 {
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }
    fn rank(&self) -> u8 {
        self.rank
    }
}
