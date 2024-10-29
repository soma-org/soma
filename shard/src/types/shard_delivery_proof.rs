use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::{digest::Digest, shard::ShardRef, shard_endorsement::ShardEndorsement};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardDeliveryProofAPI)]
pub enum ShardDeliveryProof {
    V1(ShardDeliveryProofV1),
}

#[enum_dispatch]
trait ShardDeliveryProofAPI {
    fn shard_ref(&self) -> &ShardRef;
    fn endorsement_digest(&self) -> &Digest<ShardEndorsement>;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardDeliveryProofV1 {
    shard_ref: ShardRef,
    endorsement_digest: Digest<ShardEndorsement>,
}

impl ShardDeliveryProofV1 {
    pub(crate) const fn new(
        shard_ref: ShardRef,
        endorsement_digest: Digest<ShardEndorsement>,
    ) -> Self {
        Self {
            shard_ref,
            endorsement_digest,
        }
    }
}

impl ShardDeliveryProofAPI for ShardDeliveryProofV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn endorsement_digest(&self) -> &Digest<ShardEndorsement> {
        &self.endorsement_digest
    }
}
