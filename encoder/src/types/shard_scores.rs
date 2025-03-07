use std::collections::HashSet;

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::{encoder_committee::EncoderIndex, shard_verifier::ShardAuthToken};

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
    fn scores(&self) -> Vec<Score>;
    fn unique_slots(&self) -> usize;
    fn slots(&self) -> Vec<EncoderIndex>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(crate) struct ShardScoresV1 {
    auth_token: ShardAuthToken,
    scores: Vec<ScoreV1>,
}

impl ShardScoresV1 {
    pub(crate) const fn new(auth_token: ShardAuthToken, scores: Vec<ScoreV1>) -> Self {
        Self { auth_token, scores }
    }
}

impl ShardScoresAPI for ShardScoresV1 {
    fn auth_token(&self) -> &ShardAuthToken {
        &self.auth_token
    }
    fn scores(&self) -> Vec<Score> {
        self.scores.iter().map(|s| Score::V1(*s)).collect()
    }
    fn unique_slots(&self) -> usize {
        let unique_slots: HashSet<EncoderIndex> =
            self.scores.iter().map(|score| score.slot).collect();
        unique_slots.len()
    }
    fn slots(&self) -> Vec<EncoderIndex> {
        self.scores.iter().map(|score| score.slot).collect()
    }
}

#[enum_dispatch]
pub trait ScoreAPI {
    fn slot(&self) -> EncoderIndex;
    fn rank(&self) -> u8;
}

/// Compression is the top level type. Notice that the MetadataAPI returns
/// Compression not CompressionV1
#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[enum_dispatch(ScoreAPI)]
pub enum Score {
    V1(ScoreV1),
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ScoreV1 {
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
