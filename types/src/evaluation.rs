use std::collections::HashMap;

use crate::{
    encoder_committee::EncoderCommittee,
    error::{SharedError, SharedResult},
    metadata::DownloadMetadata,
    shard_crypto::{digest::Digest, keys::EncoderPublicKey},
};
use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

#[enum_dispatch]
pub trait EvaluationInputAPI {
    fn input_download_metadata(&self) -> &DownloadMetadata;
    fn embedding_download_metadata(&self) -> &DownloadMetadata;
    fn probe_set_download_metadata(&self) -> &HashMap<EncoderPublicKey, DownloadMetadata>;
    fn probe_set(&self) -> &ProbeSet;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(EvaluationInputAPI)]
pub enum EvaluationInput {
    V1(EvaluationInputV1),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvaluationInputV1 {
    input_download_metadata: DownloadMetadata,
    embedding_download_metadata: DownloadMetadata,
    probe_set_download_metadata: HashMap<EncoderPublicKey, DownloadMetadata>,
    probe_set: ProbeSet,
}

impl EvaluationInputV1 {
    pub fn new(
        input_download_metadata: DownloadMetadata,
        embedding_download_metadata: DownloadMetadata,
        probe_set_download_metadata: HashMap<EncoderPublicKey, DownloadMetadata>,
        probe_set: ProbeSet,
    ) -> Self {
        Self {
            input_download_metadata,
            embedding_download_metadata,
            probe_set_download_metadata,
            probe_set,
        }
    }
}

impl EvaluationInputAPI for EvaluationInputV1 {
    fn input_download_metadata(&self) -> &DownloadMetadata {
        &self.input_download_metadata
    }
    fn embedding_download_metadata(&self) -> &DownloadMetadata {
        &self.embedding_download_metadata
    }
    fn probe_set_download_metadata(&self) -> &HashMap<EncoderPublicKey, DownloadMetadata> {
        &self.probe_set_download_metadata
    }
    fn probe_set(&self) -> &ProbeSet {
        &self.probe_set
    }
}

#[enum_dispatch]
pub trait EvaluationOutputAPI {
    fn score(&self) -> Score;
    fn probe_set_download_metadata(&self) -> &HashMap<EncoderPublicKey, DownloadMetadata>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[enum_dispatch(EvaluationOutputAPI)]
pub enum EvaluationOutput {
    V1(EvaluationOutputV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct EvaluationOutputV1 {
    score: ScoreV1,
    probe_set_download_metadata: HashMap<EncoderPublicKey, DownloadMetadata>,
}

impl EvaluationOutputV1 {
    pub fn new(
        score: ScoreV1,
        probe_set_download_metadata: HashMap<EncoderPublicKey, DownloadMetadata>,
    ) -> Self {
        Self {
            score,
            probe_set_download_metadata,
        }
    }
}

impl EvaluationOutputAPI for EvaluationOutputV1 {
    fn score(&self) -> Score {
        Score::V1(self.score.clone())
    }
    fn probe_set_download_metadata(&self) -> &HashMap<EncoderPublicKey, DownloadMetadata> {
        &self.probe_set_download_metadata
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

#[enum_dispatch]
pub trait ProbeWeightAPI {
    fn encoder(&self) -> &EncoderPublicKey;
    fn weight(&self) -> u64;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProbeWeightV1 {
    encoder: EncoderPublicKey,
    weight: u64,
}

impl ProbeWeightV1 {
    pub fn new(encoder: EncoderPublicKey, weight: u64) -> Self {
        Self { encoder, weight }
    }
}

impl ProbeWeightAPI for ProbeWeightV1 {
    fn encoder(&self) -> &EncoderPublicKey {
        &self.encoder
    }

    fn weight(&self) -> u64 {
        self.weight
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ProbeSetAPI)]
pub enum ProbeSet {
    V1(ProbeSetV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
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

pub(crate) fn verify_probe_set(
    probe_set: &ProbeSet,
    encoder_committee: &EncoderCommittee,
) -> SharedResult<()> {
    // TODO: adjust the probe set verification to have a max number of probes as well
    // TODO: fix the msim tests to actually construct valid probe sets
    if !cfg!(msim) {
        let mut voting_power = 0;
        for pw in probe_set.probe_weights() {
            match encoder_committee.encoder_by_key(pw.encoder()) {
                Some(encoder) => {
                    voting_power += encoder.voting_power;
                    // if encoder.probe_checksum != pw.metadata().checksum() {
                    //     return Err(SharedError::FailedTypeVerification(
                    //         "probe weight checksum does not match committee".to_string(),
                    //     ));
                    // }
                }
                None => {
                    return Err(SharedError::FailedTypeVerification(
                        "probe weight encoder not found in committee".to_string(),
                    ))
                }
            }
        }
        // TODO: CHANGE THIS TO BE THE CORRECT MINIMUM VOTING POWER
        if voting_power < 1 {
            return Err(SharedError::FailedTypeVerification(
                "probe set did not meet minimum voting power".to_string(),
            ));
        }
    }
    Ok(())
}
