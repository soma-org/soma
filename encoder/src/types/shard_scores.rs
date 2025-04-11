use std::{collections::HashSet, ops::Deref};

use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{digest::Digest, error::SharedResult, scope::Scope, signed::Signed};

use super::{
    encoder_committee::{Epoch, EvaluationEncoder, InferenceEncoder},
    shard::Shard,
    shard_verifier::ShardAuthToken,
};

/// Shard commit is the wrapper that contains the versioned shard commit. It
/// represents the encoders response to a batch of data
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ShardScoresAPI)]
pub enum ShardScores {
    V1(ShardScoresV1),
}

/// `ShardScoresAPI` is the trait that every shard commit version must implement
#[enum_dispatch]
pub(crate) trait ShardScoresAPI {
    fn auth_token(&self) -> &ShardAuthToken;
    fn evaluator(&self) -> &EvaluationEncoder;
    fn signed_score_set(&self) -> Signed<ScoreSet, min_sig::BLS12381Signature>;
    fn unique_scores(&self) -> usize;
    fn inference_encoders(&self) -> Vec<InferenceEncoder>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct ShardScoresV1 {
    auth_token: ShardAuthToken,
    evaluator: EvaluationEncoder,
    signed_score_set: Signed<ScoreSet, min_sig::BLS12381Signature>,
}

impl ShardScoresV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        evaluator: EvaluationEncoder,
        signed_score_set: Signed<ScoreSet, min_sig::BLS12381Signature>,
    ) -> Self {
        Self {
            auth_token,
            evaluator,
            signed_score_set,
        }
    }
}

impl ShardScoresAPI for ShardScoresV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn evaluator(&self) -> &EvaluationEncoder {
        &self.evaluator
    }
    fn signed_score_set(&self) -> Signed<ScoreSet, min_sig::BLS12381Signature> {
        self.signed_score_set.clone()
    }
    fn unique_scores(&self) -> usize {
        let unique_slots: &HashSet<InferenceEncoder> = &self
            .signed_score_set
            .deref()
            .scores()
            .iter()
            .map(|score| score.inference_encoder().clone())
            .collect();
        unique_slots.len()
    }
    fn inference_encoders(&self) -> Vec<InferenceEncoder> {
        self.signed_score_set
            .deref()
            .scores()
            .iter()
            .map(|score| score.inference_encoder().clone())
            .collect()
    }
}

#[enum_dispatch]
pub trait ScoreSetAPI {
    fn scores(&self) -> Vec<Score>;
}

/// Compression is the top level type. Notice that the MetadataAPI returns
/// Compression not CompressionV1
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ScoreSetAPI)]
pub enum ScoreSet {
    V1(ScoreSetV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ScoreSetV1 {
    epoch: Epoch,
    shard_ref: Digest<Shard>,
    scores: Vec<ScoreV1>,
}
impl ScoreSetV1 {
    pub fn new(epoch: Epoch, shard_ref: Digest<Shard>, mut scores: Vec<ScoreV1>) -> Self {
        scores.sort();
        Self {
            epoch,
            shard_ref,
            scores,
        }
    }
}

impl ScoreSetAPI for ScoreSetV1 {
    fn scores(&self) -> Vec<Score> {
        self.scores.iter().map(|s| Score::V1(s.clone())).collect()
    }
}

#[enum_dispatch]
pub trait ScoreAPI {
    fn inference_encoder(&self) -> &InferenceEncoder;
    fn rank(&self) -> u8;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ScoreAPI)]
pub enum Score {
    V1(ScoreV1),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ScoreV1 {
    inference_encoder: InferenceEncoder,
    rank: u8,
}
impl ScoreV1 {
    pub fn new(inference_encoder: InferenceEncoder, rank: u8) -> Self {
        Self {
            inference_encoder,
            rank,
        }
    }
}

impl ScoreAPI for ScoreV1 {
    fn inference_encoder(&self) -> &InferenceEncoder {
        &self.inference_encoder
    }
    fn rank(&self) -> u8 {
        self.rank
    }
}

pub(crate) fn verify_signed_scores(
    signed_scores: &Signed<ShardScores, min_sig::BLS12381Signature>,
    shard: &Shard,
) -> SharedResult<()> {
    if !shard.evaluation_set_contains(&signed_scores.evaluator()) {
        return Err(shared::error::SharedError::ValidationError(
            "sender is not in evaluation set".to_string(),
        ));
    }

    if signed_scores.unique_scores() != shard.inference_set_size() {
        return Err(shared::error::SharedError::ValidationError(
            "unique slots for scores does not match shard size".to_string(),
        ));
    }
    for inference_encoder in signed_scores.inference_encoders() {
        if !shard.inference_set_contains(&inference_encoder) {
            return Err(shared::error::SharedError::ValidationError(
                "score slot not in inference set".to_string(),
            ));
        }
    }

    let _ = signed_scores.verify(Scope::ShardScores, signed_scores.evaluator().inner())?;

    Ok(())
}
