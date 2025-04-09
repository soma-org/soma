use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::object::ObjectID;

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct StakingPool {
    pub id: ObjectID,
    /// Epoch when this pool became active (None = preactive)
    pub activation_epoch: Option<u64>,
    /// Epoch when deactivated (None = active)
    pub deactivation_epoch: Option<u64>,
    /// Total SOMA balance in this pool
    pub soma_balance: u64,
    /// Rewards balance
    pub rewards_pool: u64,
    /// Total pool tokens issued
    pub pool_token_balance: u64,
    /// Exchange rates by epoch
    pub exchange_rates: BTreeMap<u64, PoolTokenExchangeRate>,
    /// Pending stake awaiting processing
    pub pending_stake: u64,
    /// Pending withdrawals
    pub pending_total_soma_withdraw: u64,
    /// Pending pool token withdrawals
    pub pending_pool_token_withdraw: u64,
}

impl StakingPool {
    pub fn new() -> Self {
        Self {
            id: ObjectID::random(),
            activation_epoch: None,
            deactivation_epoch: None,
            soma_balance: 0,
            rewards_pool: 0,
            pool_token_balance: 0,
            exchange_rates: BTreeMap::new(),
            pending_stake: 0,
            pending_pool_token_withdraw: 0,
            pending_total_soma_withdraw: 0,
        }
    }

    /// Request to add stake to the staking pool
    pub fn request_add_stake(&mut self, stake: u64, stake_activation_epoch: u64) -> StakedSoma {
        assert!(stake > 0, "Stake amount must be greater than zero");
        assert!(!self.is_inactive(), "Cannot stake with inactive pool");

        // Create StakedSoma
        let staked_soma = StakedSoma::new(self.id, stake_activation_epoch, stake);

        // Update pending stake
        self.pending_stake += stake;

        staked_soma
    }

    /// Request to withdraw stake from the staking pool
    pub fn request_withdraw_stake(&mut self, staked_soma: StakedSoma, current_epoch: u64) -> u64 {
        // Validate the staking pool ID matches
        assert!(
            staked_soma.pool_id == self.id,
            "StakedSoma belongs to a different pool"
        );

        // If stake is not yet active (activation is in the future), just return principal
        if staked_soma.stake_activation_epoch > current_epoch {
            self.pending_stake -= staked_soma.principal;
            return staked_soma.principal;
        }

        // For active stake, we need to calculate rewards
        let (pool_token_amount, principal_amount) = self.withdraw_from_principal(&staked_soma);

        // Calculate rewards using the exchange rate at current epoch
        let rewards_amount =
            self.withdraw_rewards(principal_amount, pool_token_amount, current_epoch);

        let total_withdraw_amount = principal_amount + rewards_amount;

        // Update pending withdrawals
        self.pending_total_soma_withdraw += total_withdraw_amount;
        self.pending_pool_token_withdraw += pool_token_amount;

        // If pool is inactive, process withdraw immediately
        if self.is_inactive() {
            self.process_pending_stake_withdraw();
        }

        total_withdraw_amount
    }

    /// Check if the staking pool is inactive
    pub fn is_inactive(&self) -> bool {
        self.deactivation_epoch.is_some()
    }

    /// Check if the staking pool is preactive (not yet activated)
    pub fn is_preactive(&self) -> bool {
        self.activation_epoch.is_none()
    }

    /// Calculate pool tokens and principal amount when withdrawing
    pub fn withdraw_from_principal(&self, staked_soma: &StakedSoma) -> (u64, u64) {
        // Get exchange rate at staking epoch
        let exchange_rate =
            self.pool_token_exchange_rate_at_epoch(staked_soma.stake_activation_epoch);

        // Calculate pool tokens equivalent to principal
        let pool_token_amount = self.get_token_amount(&exchange_rate, staked_soma.principal);

        (pool_token_amount, staked_soma.principal)
    }

    /// Calculate and withdraw rewards
    pub fn withdraw_rewards(
        &mut self,
        principal_amount: u64,
        pool_token_amount: u64,
        epoch: u64,
    ) -> u64 {
        // Get current exchange rate
        let exchange_rate = self.pool_token_exchange_rate_at_epoch(epoch);

        // Calculate total SOMA value of the pool tokens at current exchange rate
        let total_withdraw_value = self.get_soma_amount(&exchange_rate, pool_token_amount);

        // Rewards are the difference between total value and principal
        // If total value is less than principal (which shouldn't happen in normal operation),
        // return 0 rewards to avoid underflow
        let reward_amount = if total_withdraw_value > principal_amount {
            total_withdraw_value - principal_amount
        } else {
            0
        };

        // Cap rewards at available reward pool balance
        let reward_amount = std::cmp::min(reward_amount, self.rewards_pool);

        // Deduct from rewards pool
        self.rewards_pool -= reward_amount;

        reward_amount
    }

    /// Process pending stake withdrawals
    pub fn process_pending_stake_withdraw(&mut self) {
        // Update balances based on pending withdrawals
        self.soma_balance -= self.pending_total_soma_withdraw;
        self.pool_token_balance -= self.pending_pool_token_withdraw;

        // Reset pending withdrawal amounts
        self.pending_total_soma_withdraw = 0;
        self.pending_pool_token_withdraw = 0;
    }

    /// Process pending stakes at epoch boundaries
    pub fn process_pending_stake(&mut self) {
        // Calculate the latest exchange rate based on current balances
        let latest_exchange_rate = PoolTokenExchangeRate {
            soma_amount: self.soma_balance,
            pool_token_amount: self.pool_token_balance,
        };

        // Add pending stake to soma balance
        self.soma_balance += self.pending_stake;

        // Calculate and update pool token balance
        // If pool is empty (both balances are 0), then pool tokens = soma tokens (1:1 ratio)
        if self.soma_balance == self.pending_stake && self.pool_token_balance == 0 {
            self.pool_token_balance = self.pending_stake;
        } else {
            // Otherwise calculate based on exchange rate
            self.pool_token_balance =
                self.get_token_amount(&latest_exchange_rate, self.soma_balance);
        }

        // Reset pending stake
        self.pending_stake = 0;
    }

    /// Get the exchange rate for a specific epoch
    pub fn pool_token_exchange_rate_at_epoch(&self, epoch: u64) -> PoolTokenExchangeRate {
        // If pool is preactive, return initial exchange rate (which is essentially 1:1)
        if self.is_preactive() {
            return PoolTokenExchangeRate {
                soma_amount: 0,
                pool_token_amount: 0,
            };
        }

        // Determine activation epoch (we know it's Some since pool is not preactive)
        let activation_epoch = self.activation_epoch.unwrap();

        // If requested epoch is before activation, return initial rate
        if epoch < activation_epoch {
            return PoolTokenExchangeRate {
                soma_amount: 0,
                pool_token_amount: 0,
            };
        }

        // Cap epoch at deactivation epoch if the pool is inactive
        let epoch = if let Some(deactivation_epoch) = self.deactivation_epoch {
            std::cmp::min(epoch, deactivation_epoch)
        } else {
            epoch
        };

        // Find the latest epoch that's earlier than or equal to the given epoch with an entry in the table
        // Traverse backwards from the requested epoch to the activation epoch
        let mut current_epoch = epoch;
        while current_epoch >= activation_epoch {
            if let Some(rate) = self.exchange_rates.get(&current_epoch) {
                return rate.clone();
            }
            if current_epoch == 0 {
                break;
            }
            current_epoch -= 1;
        }

        // If no rate was found, return initial rate (this should be unreachable in normal operation)
        PoolTokenExchangeRate {
            soma_amount: 0,
            pool_token_amount: 0,
        }
    }

    /// Convert pool tokens to SOMA amount
    pub fn get_soma_amount(&self, exchange_rate: &PoolTokenExchangeRate, token_amount: u64) -> u64 {
        // Handle edge cases when amounts are 0
        if exchange_rate.soma_amount == 0 || exchange_rate.pool_token_amount == 0 {
            return token_amount;
        }

        // Calculate with u128 to avoid overflow
        let res = (exchange_rate.soma_amount as u128) * (token_amount as u128)
            / (exchange_rate.pool_token_amount as u128);

        res as u64
    }

    /// Convert SOMA amount to pool tokens
    pub fn get_token_amount(&self, exchange_rate: &PoolTokenExchangeRate, soma_amount: u64) -> u64 {
        // Handle edge cases when amounts are 0
        if exchange_rate.soma_amount == 0 || exchange_rate.pool_token_amount == 0 {
            return soma_amount;
        }

        // Calculate with u128 to avoid overflow
        let res = (exchange_rate.pool_token_amount as u128) * (soma_amount as u128)
            / (exchange_rate.soma_amount as u128);

        res as u64
    }

    /// Deposit rewards into the staking pool
    pub fn deposit_rewards(&mut self, reward_amount: u64) {
        // Update SOMA balance with new rewards
        self.soma_balance += reward_amount;

        // Add to rewards pool
        self.rewards_pool += reward_amount;
    }

    /// Update exchange rates at epoch boundaries
    pub fn update_exchange_rate(&mut self, epoch: u64) {
        // Add current exchange rate to the table
        self.exchange_rates.insert(
            epoch,
            PoolTokenExchangeRate {
                soma_amount: self.soma_balance,
                pool_token_amount: self.pool_token_balance,
            },
        );
    }

    /// Process pending stakes and withdrawals at epoch boundary
    pub fn process_pending_stakes_and_withdraws(&mut self, epoch: u64) {
        // Process withdrawals first
        self.process_pending_stake_withdraw();

        // Then process new stakes
        self.process_pending_stake();

        // Finally, update exchange rate for the new epoch
        self.update_exchange_rate(epoch);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct PoolTokenExchangeRate {
    /// Amount of SOMA tokens
    pub soma_amount: u64,
    /// Amount of pool tokens
    pub pool_token_amount: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct StakedSoma {
    /// Staking pool ID
    pub pool_id: ObjectID,
    /// Epoch when stake becomes active
    pub stake_activation_epoch: u64,
    /// Principal amount staked
    pub principal: u64,
}

impl StakedSoma {
    pub fn new(pool_id: ObjectID, stake_activation_epoch: u64, principal: u64) -> Self {
        StakedSoma {
            pool_id,
            stake_activation_epoch,
            principal,
        }
    }
}
