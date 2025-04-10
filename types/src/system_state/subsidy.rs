use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct StakeSubsidy {
    /// Balance set aside for stake subsidies
    pub balance: u64,
    /// Number of times subsidies have been distributed
    pub distribution_counter: u64,
    /// Current subsidy amount per epoch
    pub current_distribution_amount: u64,
    /// Number of distributions before amount decays
    pub period_length: u64,
    /// Decay rate in basis points
    pub decrease_rate: u16,
}

impl StakeSubsidy {
    pub fn new(
        balance: u64,
        initial_distribution_amount: u64,
        period_length: u64,
        decrease_rate: u16,
    ) -> Self {
        // Rate can't be higher than 100%
        assert!(decrease_rate <= 10000, "Subsidy decrease rate too large");

        Self {
            balance,
            distribution_counter: 0,
            current_distribution_amount: initial_distribution_amount,
            period_length,
            decrease_rate,
        }
    }

    /// Calculate and withdraw subsidy for current epoch
    pub fn advance_epoch(&mut self) -> u64 {
        // Take minimum of reward amount and remaining balance
        let to_withdraw = std::cmp::min(self.current_distribution_amount, self.balance);

        // Draw down the subsidy
        self.balance -= to_withdraw;
        self.distribution_counter += 1;

        // Decrease subsidy amount at end of period
        if self.distribution_counter % self.period_length == 0 {
            let decrease_amount =
                (self.current_distribution_amount as u128) * (self.decrease_rate as u128) / 10000;
            self.current_distribution_amount -= decrease_amount as u64;
        }

        to_withdraw
    }

    /// Returns subsidy amount for current epoch
    pub fn current_epoch_subsidy_amount(&self) -> u64 {
        std::cmp::min(self.current_distribution_amount, self.balance)
    }
}
