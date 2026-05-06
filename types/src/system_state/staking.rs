// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! F1-only staking pool. Stage 9d-C5 deleted the pool-token /
//! exchange-rate machinery and the StakedSomaV1 object type — the
//! `delegations` column family is now the sole source of per-staker
//! truth, and `StakingPool::total_stake` (sum of all delegation
//! principals on the pool) drives validator voting power.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::object::ObjectID;

/// F1 index scaling factor: cumulative reward indices are stored as fixed-point
/// numbers with this denominator. 1e18 mirrors Cosmos SDK's `Dec` scale and
/// keeps rounding error well below 1 shannon for any realistic stake size.
pub const F1_INDEX_SCALE: u128 = 1_000_000_000_000_000_000;

/// Per-validator staking pool. Tracks total committed principal +
/// the F1 cumulative-reward index. There is no per-stake bookkeeping
/// here — that lives in the `delegations` column family, keyed by
/// (pool, staker).
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct StakingPool {
    pub id: ObjectID,
    /// Epoch when this pool became active (None = preactive)
    pub activation_epoch: Option<u64>,
    /// Epoch when deactivated (None = active)
    pub deactivation_epoch: Option<u64>,
    /// Sum of all delegation-row principals on this pool. Drives
    /// voting power and is the F1 fold denominator.
    pub total_stake: u64,

    // ---------------------------------------------------------------
    // F1 fee-distribution (Cosmos x/distribution shape).
    //
    // * `pool_rewards`: total unpaid reward shannons living on the
    //   pool. Increases on emission deposit, decreases on staker
    //   pay-out. This is the reward "bank" — supply conservation
    //   relies on it being explicit.
    // * `pending_fold_rewards`: shannons that have been deposited
    //   since the last fold. Drains into `cumulative_index` on the
    //   next call to `f1_fold_rewards` and clears to 0. (A separate
    //   accumulator from `pool_rewards` because the index math
    //   needs "since-last-fold" while the bank stays around to back
    //   pay-outs.)
    // * `current_period`: indexes into `cumulative_index`. Folds
    //   happen at every epoch boundary (or sooner, if a stake-set
    //   change triggers an explicit fold).
    // * `cumulative_index[p]`: monotonically-increasing total
    //   reward-per-stake from genesis through period `p`, scaled by
    //   F1_INDEX_SCALE.
    // * Pending reward for a delegator = principal *
    //   (R(current_period) − R(last_collected_period)) / scale.
    // * Commission credit is split off the top in
    //   `Validator::deposit_staker_rewards` (today the credit is
    //   applied directly to the validator's delegation row via
    //   `add_stake_principal`; `accumulated_commission` is reserved
    //   for a future "set aside without compounding" mode).
    // ---------------------------------------------------------------
    pub pool_rewards: u64,
    pub pending_fold_rewards: u128,
    pub current_period: u64,
    pub cumulative_index: BTreeMap<u64, u128>,
    pub accumulated_commission: u64,
}

impl StakingPool {
    pub fn new(id: ObjectID) -> Self {
        let mut cumulative_index = BTreeMap::new();
        cumulative_index.insert(0, 0u128);
        Self {
            id,
            activation_epoch: None,
            deactivation_epoch: None,
            total_stake: 0,
            pool_rewards: 0,
            pending_fold_rewards: 0,
            current_period: 0,
            cumulative_index,
            accumulated_commission: 0,
        }
    }

    /// Check if the staking pool is inactive
    pub fn is_inactive(&self) -> bool {
        self.deactivation_epoch.is_some()
    }

    /// Check if the staking pool is preactive (not yet activated)
    pub fn is_preactive(&self) -> bool {
        self.activation_epoch.is_none()
    }

    /// Increment total_stake. Called by AddStake when the staker's
    /// principal grows; by ChangeEpoch when the validator's
    /// commission credit lands.
    pub fn add_principal(&mut self, amount: u64) {
        self.total_stake = self.total_stake.saturating_add(amount);
    }

    /// Decrement total_stake. Called by WithdrawStake when the
    /// staker's principal shrinks. Saturating sub: an underflow
    /// here would indicate corruption upstream — the executor
    /// validates against the row's principal before mutating.
    pub fn remove_principal(&mut self, amount: u64) {
        self.total_stake = self.total_stake.saturating_sub(amount);
    }

    // ---------------------------------------------------------------
    // F1 helpers
    // ---------------------------------------------------------------

    /// Look up the cumulative reward-per-stake index at period `p`.
    /// Out-of-range queries clamp to the latest recorded period.
    pub fn f1_index_at(&self, p: u64) -> u128 {
        if let Some(v) = self.cumulative_index.get(&p) {
            return *v;
        }
        self.cumulative_index
            .range(..=p)
            .next_back()
            .map(|(_, v)| *v)
            .unwrap_or(0)
    }

    /// Compute pending reward for a delegator with `principal` whose
    /// last collection was at `last_collected_period`. The result is
    /// in shannons; the F1_INDEX_SCALE is divided out here.
    pub fn f1_pending_reward(&self, principal: u64, last_collected_period: u64) -> u64 {
        let cur = self.f1_index_at(self.current_period);
        let last = self.f1_index_at(last_collected_period);
        if cur <= last {
            return 0;
        }
        let delta = cur - last;
        ((principal as u128).saturating_mul(delta) / F1_INDEX_SCALE) as u64
    }

    /// Fold `pending_fold_rewards` into the index using `total_stake`
    /// as the divisor and advance `current_period`. Called at every
    /// epoch boundary. The shannons themselves stay parked in
    /// `pool_rewards` until a delegator claims via
    /// `f1_consume_pending_reward`.
    ///
    /// **Zero-stake invariant:** when `total_stake == 0` we cannot
    /// divide; advancing the period without distributing would
    /// strand any `pending_fold_rewards` (the SOMA stays in
    /// `pool_rewards` but the cumulative index never grows on its
    /// behalf, so no future delegator can claim it). Carry
    /// `pending_fold_rewards` forward so the next non-empty fold
    /// distributes them. The period still advances so fresh
    /// delegators starting now begin from a clean index.
    ///
    /// **Tradeoff acknowledged:** a delegator who stakes after this
    /// zero-stake fold (`last_collected_period = current_period`) is
    /// later eligible for a share of the carried rewards when they
    /// fold in a future period — even though the rewards were earned
    /// when the pool had no committed stake. We accept this in favor
    /// of avoiding stranded SOMA. Documented for callers reasoning
    /// about reward-distribution fairness.
    pub fn f1_fold_rewards(&mut self, total_stake: u64) {
        if self.pending_fold_rewards == 0 {
            // No rewards to fold; advance period so a fresh delegator
            // that joins now starts from this index.
            self.current_period += 1;
            self.cumulative_index.insert(
                self.current_period,
                self.f1_index_at(self.current_period - 1),
            );
            return;
        }
        if total_stake == 0 {
            // No divisor — carry rewards forward to the next fold.
            // Period still advances; index unchanged.
            self.current_period += 1;
            self.cumulative_index.insert(
                self.current_period,
                self.f1_index_at(self.current_period - 1),
            );
            return;
        }
        let prev = self.f1_index_at(self.current_period);
        let added = self
            .pending_fold_rewards
            .saturating_mul(F1_INDEX_SCALE)
            / (total_stake as u128);
        self.current_period += 1;
        self.cumulative_index
            .insert(self.current_period, prev.saturating_add(added));
        self.pending_fold_rewards = 0;
    }

    /// Deposit a post-commission reward amount into the F1 pool.
    /// Increases both `pool_rewards` (the bank) and
    /// `pending_fold_rewards` (the since-last-fold accumulator that
    /// the next fold drains into the cumulative index).
    pub fn f1_deposit_pool_reward(&mut self, amount: u64) {
        self.pool_rewards = self.pool_rewards.saturating_add(amount);
        self.pending_fold_rewards = self.pending_fold_rewards.saturating_add(amount as u128);
    }

    /// Debit `amount` from `pool_rewards` when a delegator collects
    /// pending rewards. Saturating sub: per-fold rounding can cause
    /// the index-derived pending to overshoot the bank by at most
    /// 1 shannon per fold, which is treated as a cap rather than a
    /// fault (the whole pool drains to 0 at the boundary).
    pub fn f1_consume_pending_reward(&mut self, amount: u64) -> u64 {
        let actual = std::cmp::min(amount, self.pool_rewards);
        self.pool_rewards -= actual;
        actual
    }
}

/// Stage 9d-C1: the value type for the F1-shaped `delegations` column
/// family. ONE row per (pool_id, staker) — never multiple. A user can
/// stake into the same validator multiple times in one epoch or across
/// many, but they always see one consolidated row showing their total
/// principal and pending reward.
///
/// `last_collected_period` is the F1 cumulative-index period at which
/// this delegation last collected its share. AddStake / WithdrawStake
/// fold pending rewards (using R(current_period) − R(last_collected))
/// to the staker's SOMA balance and then update this field. A fresh
/// delegation reads as 0 / 0 (first-touch).
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub struct Delegation {
    pub principal: u64,
    pub last_collected_period: u64,
}

impl Delegation {
    pub fn new(principal: u64, last_collected_period: u64) -> Self {
        Self { principal, last_collected_period }
    }
}
