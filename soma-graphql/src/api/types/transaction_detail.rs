// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, DateTime, Digest, SomaAddress};

/// A transaction with its decoded kind label and optional metadata.
pub struct TransactionDetail {
    pub tx_sequence_number: i64,
    pub tx_digest: Vec<u8>,
    pub kind: String,
    pub sender: Vec<u8>,
    pub epoch: i64,
    pub timestamp_ms: i64,
    pub metadata_json: Option<String>,
}

#[Object]
impl TransactionDetail {
    /// The transaction digest (base58).
    async fn digest(&self) -> Digest {
        Digest(self.tx_digest.clone())
    }

    /// The transaction sequence number.
    async fn sequence_number(&self) -> BigInt {
        BigInt(self.tx_sequence_number)
    }

    /// The decoded transaction kind label (e.g. "SubmitData", "CreateModel", "AddStake").
    async fn kind(&self) -> &str {
        &self.kind
    }

    /// The address of the transaction sender.
    async fn sender(&self) -> SomaAddress {
        SomaAddress(self.sender.clone())
    }

    /// The epoch in which this transaction was executed.
    async fn epoch(&self) -> BigInt {
        BigInt(self.epoch)
    }

    /// When this transaction was included in a checkpoint.
    async fn timestamp(&self) -> DateTime {
        DateTime(self.timestamp_ms)
    }

    /// Kind-specific metadata as JSON (e.g. target_id, model_id, amount).
    async fn metadata_json(&self) -> Option<&str> {
        self.metadata_json.as_deref()
    }
}
