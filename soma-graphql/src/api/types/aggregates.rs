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

/// Aggregate score statistics for filled targets at a given epoch.
pub struct TargetScoreAggregates {
    pub avg_loss_score: Option<f64>,
    pub avg_distance_score: Option<f64>,
    pub total_data_size: i64,
    pub filled_count: i64,
}

#[Object]
impl TargetScoreAggregates {
    /// Average loss score across filled targets.
    async fn avg_loss_score(&self) -> Option<f64> {
        self.avg_loss_score
    }

    /// Average distance score across filled targets.
    async fn avg_distance_score(&self) -> Option<f64> {
        self.avg_distance_score
    }

    /// Total data size across all filled submissions (bytes).
    async fn total_data_size(&self) -> BigInt {
        BigInt(self.total_data_size)
    }

    /// Number of filled targets.
    async fn filled_count(&self) -> i32 {
        self.filled_count as i32
    }
}

/// Aggregate statistics for a data submitter.
pub struct SubmitterStats {
    pub submitter: Vec<u8>,
    pub target_count: i64,
    pub avg_distance_score: Option<f64>,
    pub avg_loss_score: Option<f64>,
    pub total_reward: i64,
    pub total_data_size: i64,
}

#[Object]
impl SubmitterStats {
    /// Submitter address.
    async fn submitter(&self) -> crate::api::scalars::SomaAddress {
        crate::api::scalars::SomaAddress(self.submitter.clone())
    }

    /// Number of targets filled by this submitter.
    async fn target_count(&self) -> i32 {
        self.target_count as i32
    }

    /// Average distance score across filled targets.
    async fn avg_distance_score(&self) -> Option<f64> {
        self.avg_distance_score
    }

    /// Average loss score across filled targets.
    async fn avg_loss_score(&self) -> Option<f64> {
        self.avg_loss_score
    }

    /// Total reward earned (shannons).
    async fn total_reward(&self) -> BigInt {
        BigInt(self.total_reward)
    }

    /// Total data size submitted (bytes).
    async fn total_data_size(&self) -> BigInt {
        BigInt(self.total_data_size)
    }
}

/// Aggregate statistics for a model on the leaderboard.
pub struct ModelStats {
    pub model_id: Vec<u8>,
    pub targets_won: i64,
    pub targets_assigned: i64,
    pub avg_distance_score: Option<f64>,
    pub avg_loss_score: Option<f64>,
    pub total_reward: i64,
    pub total_data_size: i64,
}

#[Object]
impl ModelStats {
    /// Model ID (hex address).
    async fn model_id(&self) -> crate::api::scalars::SomaAddress {
        crate::api::scalars::SomaAddress(self.model_id.clone())
    }

    /// Number of targets won (filled) by this model.
    async fn targets_won(&self) -> i32 {
        self.targets_won as i32
    }

    /// Number of targets this model was assigned to.
    async fn targets_assigned(&self) -> i32 {
        self.targets_assigned as i32
    }

    /// Win rate: targets won / targets assigned (0.0–1.0). Null if never assigned.
    async fn win_rate(&self) -> Option<f64> {
        if self.targets_assigned > 0 {
            Some(self.targets_won as f64 / self.targets_assigned as f64)
        } else {
            None
        }
    }

    /// Average distance score across won targets.
    async fn avg_distance_score(&self) -> Option<f64> {
        self.avg_distance_score
    }

    /// Average loss score across won targets.
    async fn avg_loss_score(&self) -> Option<f64> {
        self.avg_loss_score
    }

    /// Total reward earned by this model (shannons).
    async fn total_reward(&self) -> BigInt {
        BigInt(self.total_reward)
    }

    /// Total data size across won submissions (bytes).
    async fn total_data_size(&self) -> BigInt {
        BigInt(self.total_data_size)
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
