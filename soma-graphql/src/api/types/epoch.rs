// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, DateTime};

/// An epoch in the Soma network.
pub struct Epoch {
    // From kv_epoch_starts
    pub epoch: i64,
    pub protocol_version: i64,
    pub cp_lo: i64,
    pub start_timestamp_ms: i64,
    pub reference_gas_price: i64,
    pub system_state_bcs: Vec<u8>,
    // From kv_epoch_ends (optional — current epoch has no end)
    pub cp_hi: Option<i64>,
    pub tx_hi: Option<i64>,
    pub end_timestamp_ms: Option<i64>,
    pub safe_mode: Option<bool>,
    pub total_stake: Option<i64>,
    pub total_gas_fees: Option<i64>,
}

#[Object]
impl Epoch {
    /// The epoch number.
    async fn epoch_id(&self) -> BigInt {
        BigInt(self.epoch)
    }

    /// The protocol version active during this epoch.
    async fn protocol_version(&self) -> BigInt {
        BigInt(self.protocol_version)
    }

    /// The first checkpoint in this epoch.
    async fn start_checkpoint(&self) -> BigInt {
        BigInt(self.cp_lo)
    }

    /// The last checkpoint in this epoch (null if epoch is ongoing).
    async fn end_checkpoint(&self) -> Option<BigInt> {
        self.cp_hi.map(BigInt)
    }

    /// When this epoch started.
    async fn start_timestamp(&self) -> DateTime {
        DateTime(self.start_timestamp_ms)
    }

    /// When this epoch ended (null if ongoing).
    async fn end_timestamp(&self) -> Option<DateTime> {
        self.end_timestamp_ms.map(DateTime)
    }

    /// The reference gas price for this epoch.
    async fn reference_gas_price(&self) -> BigInt {
        BigInt(self.reference_gas_price)
    }

    /// Whether the epoch ended in safe mode.
    async fn safe_mode(&self) -> Option<bool> {
        self.safe_mode
    }

    /// Total stake in this epoch (null if not yet ended or unavailable).
    async fn total_stake(&self) -> Option<BigInt> {
        self.total_stake.map(BigInt)
    }

    /// Total gas fees collected in this epoch.
    async fn total_gas_fees(&self) -> Option<BigInt> {
        self.total_gas_fees.map(BigInt)
    }

    /// The last transaction sequence number in this epoch.
    async fn tx_hi(&self) -> Option<BigInt> {
        self.tx_hi.map(BigInt)
    }

    /// BCS-serialized system state at epoch start.
    async fn system_state_bcs(&self) -> Base64 {
        Base64(self.system_state_bcs.clone())
    }
}
