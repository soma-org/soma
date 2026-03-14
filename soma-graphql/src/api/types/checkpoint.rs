// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, DateTime, Digest};

/// A checkpoint in the Soma network.
pub struct Checkpoint {
    pub sequence_number: i64,
    pub checkpoint_summary_bcs: Vec<u8>,
    pub checkpoint_contents_bcs: Vec<u8>,
    pub validator_signatures_bcs: Vec<u8>,
    // Denormalized from cp_sequence_numbers
    pub epoch: Option<i64>,
    pub tx_lo: Option<i64>,
    pub timestamp_ms: Option<i64>,
}

#[Object]
impl Checkpoint {
    /// The checkpoint's sequence number.
    async fn sequence_number(&self) -> BigInt {
        BigInt(self.sequence_number)
    }

    /// The epoch this checkpoint belongs to.
    async fn epoch(&self) -> Option<BigInt> {
        self.epoch.map(BigInt)
    }

    /// The first transaction sequence number in this checkpoint.
    async fn tx_lo(&self) -> Option<BigInt> {
        self.tx_lo.map(BigInt)
    }

    /// The timestamp of this checkpoint in milliseconds.
    async fn timestamp(&self) -> Option<DateTime> {
        self.timestamp_ms.map(DateTime)
    }

    /// BCS-serialized checkpoint summary.
    async fn checkpoint_summary_bcs(&self) -> Base64 {
        Base64(self.checkpoint_summary_bcs.clone())
    }

    /// BCS-serialized checkpoint contents.
    async fn checkpoint_contents_bcs(&self) -> Base64 {
        Base64(self.checkpoint_contents_bcs.clone())
    }

    /// BCS-serialized validator signatures.
    async fn validator_signatures_bcs(&self) -> Base64 {
        Base64(self.validator_signatures_bcs.clone())
    }
}
