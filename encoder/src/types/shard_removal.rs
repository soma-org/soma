use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::network_committee::NetworkingIndex;

use super::shard::ShardRef;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardRemovalAPI)]
pub enum ShardRemoval {
    V1(ShardRemovalV1),
}

#[enum_dispatch]
trait ShardRemovalAPI {
    fn shard_ref(&self) -> &ShardRef;
    fn shard_member(&self) -> &NetworkingIndex;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardRemovalV1 {
    shard_ref: ShardRef,
    shard_member: NetworkingIndex,
}

impl ShardRemovalV1 {
    pub(crate) const fn new(shard_ref: ShardRef, shard_member: NetworkingIndex) -> Self {
        Self {
            shard_ref,
            shard_member,
        }
    }
}

impl ShardRemovalAPI for ShardRemovalV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn shard_member(&self) -> &NetworkingIndex {
        &self.shard_member
    }
}
