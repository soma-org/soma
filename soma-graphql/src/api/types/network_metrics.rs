// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::BigInt;

/// Network-wide metrics.
pub struct NetworkMetrics {
    pub tps: Option<f64>,
    pub total_transactions: i64,
    pub total_checkpoints: i64,
    pub total_validators: i64,
}

#[Object]
impl NetworkMetrics {
    /// Transactions per second (computed from recent checkpoints).
    /// May be null if checkpoint timestamps are unavailable.
    async fn tps(&self) -> Option<f64> {
        self.tps
    }

    /// Total number of transactions processed.
    async fn total_transactions(&self) -> BigInt {
        BigInt(self.total_transactions)
    }

    /// Total number of checkpoints.
    async fn total_checkpoints(&self) -> BigInt {
        BigInt(self.total_checkpoints)
    }

    /// Total number of validators (at latest epoch).
    async fn total_validators(&self) -> i32 {
        self.total_validators as i32
    }
}
