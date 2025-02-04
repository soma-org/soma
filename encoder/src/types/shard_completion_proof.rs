use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::shard::ShardRef;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardCompletionProofAPI)]
pub enum ShardCompletionProof {
    V1(ShardCompletionProofV1),
}

#[enum_dispatch]
trait ShardCompletionProofAPI {
    fn shard_ref(&self) -> &ShardRef;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardCompletionProofV1 {
    shard_ref: ShardRef,
}

impl ShardCompletionProofV1 {
    pub(crate) const fn new(shard_ref: ShardRef) -> Self {
        Self { shard_ref }
    }
}

impl ShardCompletionProofAPI for ShardCompletionProofV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
}
