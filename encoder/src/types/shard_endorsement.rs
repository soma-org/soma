use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::metadata::Metadata;

use crate::types::{score::Score, shard::ShardRef};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardEndorsementAPI)]
pub enum ShardEndorsement {
    V1(ShardEndorsementV1),
}

#[enum_dispatch]
pub trait ShardEndorsementAPI {
    fn scores(&self) -> &[Score];
    fn data(&self) -> &Metadata;
    fn shard_ref(&self) -> &ShardRef;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardEndorsementV1 {
    scores: Vec<Score>,
    data: Metadata,
    shard_ref: ShardRef,
}

impl ShardEndorsementV1 {
    pub(crate) fn new(scores: Vec<Score>, data: Metadata, shard_ref: ShardRef) -> Self {
        Self {
            scores,
            data,
            shard_ref,
        }
    }
}

impl ShardEndorsementAPI for ShardEndorsementV1 {
    fn scores(&self) -> &[Score] {
        &self.scores
    }
    fn data(&self) -> &Metadata {
        &self.data
    }

    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
}
