// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, DateTime, Digest};

/// A transaction on the Soma network.
pub struct Transaction {
    pub tx_digest: Vec<u8>,
    pub cp_sequence_number: i64,
    pub timestamp_ms: i64,
    pub raw_transaction_bcs: Vec<u8>,
    pub raw_effects_bcs: Vec<u8>,
    pub user_signatures_bcs: Vec<u8>,
}

#[Object]
impl Transaction {
    /// The transaction digest (base58).
    async fn digest(&self) -> Digest {
        Digest(self.tx_digest.clone())
    }

    /// The checkpoint that included this transaction.
    async fn checkpoint_sequence_number(&self) -> BigInt {
        BigInt(self.cp_sequence_number)
    }

    /// When this transaction was included in a checkpoint.
    async fn timestamp(&self) -> DateTime {
        DateTime(self.timestamp_ms)
    }

    /// BCS-serialized transaction data.
    async fn raw_transaction_bcs(&self) -> Base64 {
        Base64(self.raw_transaction_bcs.clone())
    }

    /// BCS-serialized transaction effects.
    async fn raw_effects_bcs(&self) -> Base64 {
        Base64(self.raw_effects_bcs.clone())
    }

    /// BCS-serialized user signatures.
    async fn user_signatures_bcs(&self) -> Base64 {
        Base64(self.user_signatures_bcs.clone())
    }
}
