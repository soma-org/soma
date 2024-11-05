use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::shard::ShardRef;
use crate::types::network_committee::NetworkingIndex;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardSlotsAPI)]
pub enum ShardSlots {
    V1(ShardSlotsV1),
}

#[enum_dispatch]
pub(crate) trait ShardSlotsAPI {
    fn shard_ref(&self) -> &ShardRef;
    fn shard_members(&self) -> &Vec<NetworkingIndex>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardSlotsV1 {
    shard_ref: ShardRef,
    shard_members: Vec<NetworkingIndex>,
}

impl ShardSlotsV1 {
    pub(crate) const fn new(shard_ref: ShardRef, shard_members: Vec<NetworkingIndex>) -> Self {
        Self {
            shard_ref,
            shard_members,
        }
    }
}

impl ShardSlotsAPI for ShardSlotsV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn shard_members(&self) -> &Vec<NetworkingIndex> {
        &self.shard_members
    }
}
