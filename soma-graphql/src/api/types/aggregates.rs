// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::BigInt;

/// Aggregate statistics for models at a given epoch.
pub struct ModelAggregates {
    pub total_count: i64,
    pub total_stake: i64,
    pub avg_stake: f64,
    pub active_count: i64,
}

#[Object]
impl ModelAggregates {
    /// Total number of models.
    async fn total_count(&self) -> i32 {
        self.total_count as i32
    }

    /// Sum of all model stakes (shannons).
    async fn total_stake(&self) -> BigInt {
        BigInt(self.total_stake)
    }

    /// Average stake per model (shannons).
    async fn avg_stake(&self) -> f64 {
        self.avg_stake
    }

    /// Number of active models.
    async fn active_count(&self) -> i32 {
        self.active_count as i32
    }
}

/// Aggregate statistics for targets at a given epoch.
pub struct TargetAggregates {
    pub total_count: i64,
    pub open_count: i64,
    pub filled_count: i64,
    pub claimed_count: i64,
    pub total_reward_pool: i64,
}

#[Object]
impl TargetAggregates {
    /// Total number of targets.
    async fn total_count(&self) -> i32 {
        self.total_count as i32
    }

    /// Number of open targets.
    async fn open_count(&self) -> i32 {
        self.open_count as i32
    }

    /// Number of filled targets.
    async fn filled_count(&self) -> i32 {
        self.filled_count as i32
    }

    /// Number of claimed targets.
    async fn claimed_count(&self) -> i32 {
        self.claimed_count as i32
    }

    /// Sum of all target reward pools (shannons).
    async fn total_reward_pool(&self) -> BigInt {
        BigInt(self.total_reward_pool)
    }
}

/// Aggregate statistics for rewards at a given epoch.
pub struct RewardAggregates {
    pub total_count: i64,
    pub total_amount: i64,
}

#[Object]
impl RewardAggregates {
    /// Total number of reward claims.
    async fn total_count(&self) -> i32 {
        self.total_count as i32
    }

    /// Sum of all balance change amounts (shannons).
    async fn total_amount(&self) -> BigInt {
        BigInt(self.total_amount)
    }
}
