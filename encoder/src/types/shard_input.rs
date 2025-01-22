

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::{metadata::Metadata, transaction_certificate::TransactionCertificate};

use super::shard::ShardSecret;


#[derive(Debug, Clone, Deserialize, Serialize)]
#[enum_dispatch(ShardInputAPI)]
pub enum ShardInput {
    V1(ShardInputV1),
}

#[enum_dispatch]
pub trait ShardInputAPI {
    fn transaction_certificate(&self) -> &TransactionCertificate;
    fn shard_secret(&self) -> &ShardSecret;
    fn data(&self) -> &Metadata;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ShardInputV1 {
    transaction_certificate: TransactionCertificate,
    shard_secret: ShardSecret,
    data: Metadata,
}

impl ShardInputV1 {
    pub(crate) fn new(
        transaction_certificate: TransactionCertificate,
        shard_secret: ShardSecret,
        data: Metadata,
    ) -> Self {
        Self {
            transaction_certificate,
            shard_secret,
            data,
        }
    }
}

impl ShardInputAPI for ShardInputV1 {
    fn transaction_certificate(&self) -> &TransactionCertificate {
        &self.transaction_certificate
    }
    fn shard_secret(&self) -> &ShardSecret {
        &self.shard_secret
    }
    fn data(&self) -> &Metadata {
        &self.data
    }
}
