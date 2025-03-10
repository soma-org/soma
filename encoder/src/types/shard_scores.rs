use std::{collections::HashSet, ops::Deref};

use enum_dispatch::enum_dispatch;
use fastcrypto::bls12381::min_sig;
use serde::{Deserialize, Serialize};
use shared::{digest::Digest, signed::Signed};

use super::{
    encoder_committee::{EncoderIndex, Epoch},
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
    fn evaluator(&self) -> EncoderIndex;
    fn signed_score_set(&self) -> Signed<ScoreSet, min_sig::BLS12381Signature>;
    fn unique_slots(&self) -> usize;
    fn slots(&self) -> Vec<EncoderIndex>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct ShardScoresV1 {
    auth_token: ShardAuthToken,
    evaluator: EncoderIndex,
    signed_score_set: Signed<ScoreSet, min_sig::BLS12381Signature>,
}

impl ShardScoresV1 {
    pub(crate) const fn new(
        auth_token: ShardAuthToken,
        evaluator: EncoderIndex,
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
    fn evaluator(&self) -> EncoderIndex {
        self.evaluator
    }
    fn signed_score_set(&self) -> Signed<ScoreSet, min_sig::BLS12381Signature> {
        self.signed_score_set.clone()
    }
    fn unique_slots(&self) -> usize {
        let unique_slots: &HashSet<EncoderIndex> = &self
            .signed_score_set
            .deref()
            .scores()
            .iter()
            .map(|score| score.slot())
            .collect();
        unique_slots.len()
    }
    fn slots(&self) -> Vec<EncoderIndex> {
        self.signed_score_set
            .deref()
            .scores()
            .iter()
            .map(|score| score.slot())
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
        self.scores.iter().map(|s| Score::V1(*s)).collect()
    }
}

#[enum_dispatch]
pub trait ScoreAPI {
    fn slot(&self) -> EncoderIndex;
    fn rank(&self) -> u8;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ScoreAPI)]
pub enum Score {
    V1(ScoreV1),
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ScoreV1 {
    slot: EncoderIndex,
    rank: u8,
}
impl ScoreV1 {
    pub fn new(slot: EncoderIndex, rank: u8) -> Self {
        Self { slot, rank }
    }
}

impl ScoreAPI for ScoreV1 {
    fn slot(&self) -> EncoderIndex {
        self.slot
    }
    fn rank(&self) -> u8 {
        self.rank
    }
}
