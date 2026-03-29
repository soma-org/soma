// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::BigInt;

/// Epoch-level system state: emission pool, marketplace, and safe mode.
pub struct EpochState {
    pub epoch: i64,
    pub emission_balance: i64,
    pub emission_per_epoch: i64,
    pub distribution_counter: i64,
    pub period_length: i64,
    pub decrease_rate: i32,
    pub protocol_fund_balance: i64,
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

    /// Current emission amount per epoch (shannons). Decays over time.
    async fn emission_per_epoch(&self) -> BigInt {
        BigInt(self.emission_per_epoch)
    }

    /// Number of epochs emissions have been distributed.
    async fn distribution_counter(&self) -> BigInt {
        BigInt(self.distribution_counter)
    }

    /// Number of epochs per decay period.
    async fn period_length(&self) -> BigInt {
        BigInt(self.period_length)
    }

    /// Decay rate in basis points (e.g. 1000 = 10% decrease per period).
    async fn decrease_rate(&self) -> i32 {
        self.decrease_rate
    }

    /// Accumulated USDC from marketplace value fees (microdollars).
    async fn protocol_fund_balance(&self) -> BigInt {
        BigInt(self.protocol_fund_balance)
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
