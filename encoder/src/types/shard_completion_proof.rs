use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::transaction_certificate::TransactionCertificate;

use super::shard::ShardRef;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[enum_dispatch(ShardCompletionProofAPI)]
pub enum ShardCompletionProof {
    V1(ShardCompletionProofV1),
}

#[enum_dispatch]
trait ShardCompletionProofAPI {
    fn shard_ref(&self) -> &ShardRef;
    fn transaction_certificate(&self) -> &TransactionCertificate;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ShardCompletionProofV1 {
    shard_ref: ShardRef,
    transaction_certificate: TransactionCertificate,
}

impl ShardCompletionProofV1 {
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

impl ShardCompletionProofAPI for ShardCompletionProofV1 {
    fn shard_ref(&self) -> &ShardRef {
        &self.shard_ref
    }
    fn transaction_certificate(&self) -> &TransactionCertificate {
        &self.transaction_certificate
    }
}
