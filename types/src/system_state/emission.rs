// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// EmissionPool manages geometric step-decay emissions from genesis allocation.
///
/// Each epoch: emit `min(current_distribution_amount, balance)`.
/// Every `period_length` epochs, reduce `current_distribution_amount` by `decrease_rate` bps.
///
/// This creates geometric decay — high early emissions tapering off asymptotically.
/// Modeled after Sui's StakeSubsidy.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct EmissionPool {
    /// Remaining balance from genesis allocation
    pub balance: u64,

    /// Count of epochs emissions have been distributed
    pub distribution_counter: u64,

    /// Current emission amount per epoch (decays over time)
    pub current_distribution_amount: u64,

    /// Number of epochs per decay period
    pub period_length: u64,

    /// Decay rate in basis points (e.g., 1000 = 10% decrease per period)
    pub decrease_rate: u16,
}

impl EmissionPool {
    pub fn new(
        balance: u64,
        initial_distribution_amount: u64,
        period_length: u64,
        decrease_rate: u16,
    ) -> Self {
        Self {
            balance,
            distribution_counter: 0,
            current_distribution_amount: initial_distribution_amount,
            period_length,
            decrease_rate,
        }
    }

    /// Withdraw this epoch's emission and apply decay if at period boundary.
    /// Returns the emission amount for this epoch.
    pub fn advance_epoch(&mut self) -> u64 {
        let emission = std::cmp::min(self.current_distribution_amount, self.balance);
        self.balance -= emission;
        self.distribution_counter += 1;

        // Apply decay at period boundary
        if self.period_length > 0
            && self.distribution_counter % self.period_length == 0
            && self.decrease_rate > 0
        {
            let decrease = (self.current_distribution_amount as u128)
                * (self.decrease_rate as u128)
                / 10_000u128;
            self.current_distribution_amount =
                self.current_distribution_amount.saturating_sub(decrease as u64);
        }

        emission
    }

    /// Whether the pool still has emissions
    pub fn is_emitting(&self) -> bool {
        self.balance > 0 && self.current_distribution_amount > 0
    }

    pub fn current_epoch_emission(&self) -> u64 {
        std::cmp::min(self.current_distribution_amount, self.balance)
    }
}
