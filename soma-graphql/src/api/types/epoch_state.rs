// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::BigInt;

/// Epoch-level system state: emission pool, target state, and safe mode.
pub struct EpochState {
    pub epoch: i64,
    pub emission_balance: i64,
    pub emission_per_epoch: i64,
    pub distance_threshold: f64,
    pub targets_generated_this_epoch: i64,
    pub hits_this_epoch: i64,
    pub hits_ema: i64,
    pub reward_per_target: i64,
    pub safe_mode: bool,
    pub safe_mode_accumulated_fees: i64,
    pub safe_mode_accumulated_emissions: i64,
}

#[Object]
impl EpochState {
    /// The epoch number.
    async fn epoch(&self) -> BigInt {
        BigInt(self.epoch)
    }

    /// Remaining balance in the emission pool (shannons).
    async fn emission_balance(&self) -> BigInt {
        BigInt(self.emission_balance)
    }

    /// Fixed emission amount per epoch (shannons).
    async fn emission_per_epoch(&self) -> BigInt {
        BigInt(self.emission_per_epoch)
    }

    /// Current distance threshold for new targets.
    async fn distance_threshold(&self) -> f64 {
        self.distance_threshold
    }

    /// Number of targets generated this epoch.
    async fn targets_generated_this_epoch(&self) -> BigInt {
        BigInt(self.targets_generated_this_epoch)
    }

    /// Number of successful hits (filled targets) this epoch.
    async fn hits_this_epoch(&self) -> BigInt {
        BigInt(self.hits_this_epoch)
    }

    /// Exponential moving average of hits per epoch.
    async fn hits_ema(&self) -> BigInt {
        BigInt(self.hits_ema)
    }

    /// Reward per target for the current epoch (shannons).
    async fn reward_per_target(&self) -> BigInt {
        BigInt(self.reward_per_target)
    }

    /// Whether the system is in safe mode.
    async fn safe_mode(&self) -> bool {
        self.safe_mode
    }

    /// Fees accumulated during safe mode epochs (shannons).
    async fn safe_mode_accumulated_fees(&self) -> BigInt {
        BigInt(self.safe_mode_accumulated_fees)
    }

    /// Emissions accumulated during safe mode epochs (shannons).
    async fn safe_mode_accumulated_emissions(&self) -> BigInt {
        BigInt(self.safe_mode_accumulated_emissions)
    }
}
