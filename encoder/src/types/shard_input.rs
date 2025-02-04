use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use shared::metadata::{Metadata, MetadataCommitment};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[enum_dispatch(ShardInputAPI)]
pub enum ShardInput {
    V1(ShardInputV1),
}

#[enum_dispatch]
pub trait ShardInputAPI {
    // fn transaction_certificate(&self) -> &TransactionCertificate;
    fn commitment(&self) -> &MetadataCommitment;
    fn data(&self) -> &Metadata;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ShardInputV1 {
    // transaction_certificate: TransactionCertificate,
    commitment: MetadataCommitment,
    data: Metadata,
}

impl ShardInputV1 {
    pub(crate) fn new(commitment: MetadataCommitment, data: Metadata) -> Self {
        Self {
            // transaction_certificate,
            commitment,
            data,
        }
    }
}

impl ShardInputAPI for ShardInputV1 {
    // fn transaction_certificate(&self) -> &TransactionCertificate {
    //     &self.transaction_certificate
    // }
    fn commitment(&self) -> &MetadataCommitment {
        &self.commitment
    }
    fn data(&self) -> &Metadata {
        &self.data
    }
}
