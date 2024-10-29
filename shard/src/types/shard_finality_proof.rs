use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};

use super::{shard::ShardRef, transaction_certificate::TransactionCertificate};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardFinalityProofAPI)]
pub enum ShardFinalityProof {
    V1(ShardFinalityProofV1),
}

#[enum_dispatch]
trait ShardFinalityProofAPI {
    fn shard_ref(&self) -> &ShardRef;
    fn transaction_certificate(&self) -> &TransactionCertificate;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardFinalityProofV1 {
    shard_ref: ShardRef,
    transaction_certificate: TransactionCertificate,
}

impl ShardFinalityProofV1 {
    pub(crate) const fn new(
        shard_ref: ShardRef,
        transaction_certificate: TransactionCertificate,
    ) -> Self {
        Self {
            shard_ref,
            transaction_certificate,
        }
    }
}

impl ShardFinalityProofAPI for ShardFinalityProofV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn transaction_certificate(&self) -> &TransactionCertificate {
        &self.transaction_certificate
    }
}
