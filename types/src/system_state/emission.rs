use serde::{Deserialize, Serialize};

/// EmissionPool manages fixed per-epoch emissions from genesis allocation.
/// Fee redistribution is handled separately and passed in during advance_epoch.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct EmissionPool {
    /// Remaining balance from genesis allocation
    pub balance: u64,

    /// Fixed emission amount per epoch
    pub emission_per_epoch: u64,
}

impl EmissionPool {
    pub fn new(balance: u64, emission_per_epoch: u64) -> Self {
        Self {
            balance,
            emission_per_epoch,
        }
    }

    /// Withdraw this epoch's emission (returns 0 when pool is drained)
    pub fn advance_epoch(&mut self) -> u64 {
        let emission = std::cmp::min(self.emission_per_epoch, self.balance);
        self.balance -= emission;
        emission
    }

    /// Epochs remaining until pool is drained
    pub fn epochs_remaining(&self) -> u64 {
        if self.emission_per_epoch == 0 {
            u64::MAX
        } else {
            self.balance / self.emission_per_epoch
        }
    }

    /// Whether the pool still has emissions
    pub fn is_emitting(&self) -> bool {
        self.balance > 0
    }

    pub fn current_epoch_emission(&self) -> u64 {
        std::cmp::min(self.emission_per_epoch, self.balance)
    }
}
