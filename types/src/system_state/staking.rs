// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Auto-compound staking pool with a multiplicative
//! cumulative-index (exchange rate) reward model.
//!
//! Mirrors Sui's `staking_pool.move` semantics with two
//! simplifications:
//!
//! 1. **Single-row delegation.** One [`Delegation`] per `(pool_id,
//!    staker)`, not one object per stake action.
//! 2. **Cumulative-index reward math.** Instead of explicit pool
//!    tokens with an epoch-keyed exchange-rate table, we track a
//!    single multiplicative `cumulative_index`. The index is exactly
//!    the exchange rate `(active_stake / total_implicit_pool_tokens)`
//!    scaled by [`F1_INDEX_SCALE`]. A staker's compound at any time
//!    is `principal × current_index / index_at_last_collect`.
//!    O(1) read/write on every operation.
//!
//! ## Why multiplicative, not additive?
//!
//! An additive cumulative-index (Cosmos SDK x/distribution F1) is
//! correct only when stakers explicitly claim each epoch — the math
//! treats `principal` as constant across the claim window. For
//! auto-compound — where `principal` itself is supposed to grow each
//! epoch — additive math under-rewards stakers who stay settled-stale
//! across multiple boundaries. The multiplicative form
//! `principal × current_index / index_at_last_collect` captures the
//! compound effect correctly with a lazy O(1) settle.
//!
//! Equivalence to Sui's pool tokens: if alice owns implicit pool
//! tokens `P_t`, then `P_t × exchange_rate = principal`. Over time
//! the rate grows, and her balance grows proportionally — same
//! economics as `principal × current_index / index_at_last_collect`.
//!
//! ## Design properties
//!
//! - **Auto-compound.** Rewards become stake on deposit; existing
//!   stakers' [`Delegation::principal`] grows on next [`auto_settle`]
//!   call. No separate reward bank — the reward shannons live in
//!   [`StakingPool::active_stake`] until withdrawn.
//! - **Mid-epoch addition doesn't earn current epoch.** New stake
//!   goes to the pending bucket on both pool and delegation. It
//!   promotes into [`active_stake`][`StakingPool::active_stake`] at
//!   the next epoch boundary.
//! - **Mid-epoch withdrawal forfeits current-epoch share.** The
//!   withdrawn principal decrements
//!   [`active_stake`][`StakingPool::active_stake`] immediately; the
//!   smaller divisor at the next deposit redistributes the forfeited
//!   share to remaining stakers — equivalent to Sui's pool-token /
//!   exchange-rate redistribution.
//! - **Voting power frozen mid-epoch.** Set only at epoch boundary
//!   based on post-boundary
//!   [`active_stake`][`StakingPool::active_stake`].
//!
//! ## Invariants
//!
//! `pool.active_stake == sum_settled(delegation.principal) +
//! unclaimed_compound` across all delegations on the pool, where
//! `unclaimed_compound` catches up as stakers call [`auto_settle`].
//! After every staker has settled at the same `current_epoch`,
//! `sum(delegation.principal) == active_stake` modulo integer-
//! division rounding (bounded by ~1 shannon per delegation).

use serde::{Deserialize, Serialize};

use crate::object::ObjectID;

/// Scaling factor for the multiplicative cumulative index. The index
/// represents the pool's exchange rate (1.0 = no compound yet) as a
/// fixed-point number; 1e18 keeps rounding error well below 1 shannon
/// for realistic stake sizes and matches Cosmos SDK's `Dec` scale.
pub const F1_INDEX_SCALE: u128 = 1_000_000_000_000_000_000;

/// Per-validator staking pool. Tracks active + pending stake and the
/// F1 cumulative reward index. Per-staker bookkeeping lives on
/// [`Delegation`] rows keyed by `(pool_id, staker)`.
#[derive(Debug, Serialize, Deserialize, Clone, Eq, PartialEq, Hash)]
pub struct StakingPool {
    pub id: ObjectID,

    /// Epoch when this pool became active. `None` while preactive
    /// (validator candidate not yet in the active set).
    pub activation_epoch: Option<u64>,

    /// Epoch when deactivated. `None` while active.
    pub deactivation_epoch: Option<u64>,

    /// Total stake earning rewards in the current epoch. This is the
    /// F1 fold divisor.
    ///
    /// Updated:
    /// - At epoch boundary: grows by deposited rewards (auto-compound)
    ///   and promoted [`pending_active_stake`].
    /// - On `WithdrawStake` mid-epoch: decrements immediately
    ///   (forfeiting current-epoch share to remaining stakers via the
    ///   smaller divisor).
    ///
    /// Includes pre-claim auto-compounded rewards. Tracks the pool's
    /// claim on user balances; equal to
    /// `sum(delegation.principal) + unclaimed_compound`.
    pub active_stake: u64,

    /// Stake added during the current epoch. Does NOT count toward
    /// [`active_stake`] (the fold divisor) or earn current-epoch
    /// rewards. Promoted into [`active_stake`] at the next epoch
    /// boundary.
    pub pending_active_stake: u64,

    /// Current multiplicative cumulative index, equivalent to the
    /// pool's exchange rate scaled by [`F1_INDEX_SCALE`]. Starts at
    /// `F1_INDEX_SCALE` (= 1.0). Advances at every reward deposit in
    /// [`Self::deposit_staker_rewards`].
    ///
    /// Compound for a delegation row is computed as
    /// `principal × cumulative_index / index_at_last_collect` — i.e.
    /// the staker's principal grows by the same factor as the pool's
    /// exchange rate.
    pub cumulative_index: u128,

    /// Snapshot of [`cumulative_index`] right after each epoch's fold.
    /// `index_history[i]` = `cumulative_index` at the start of epoch
    /// `(activation_epoch.unwrap_or(0) + i + 1)`, i.e. after the fold
    /// for the previous epoch.
    ///
    /// Used by [`auto_settle`] to look up the baseline index for
    /// delegation rows whose pending stake matured before being
    /// claimed. O(1) random access by relative epoch index.
    pub index_history: Vec<u128>,

    /// Validator's commission rate in basis points (1 bp = 0.01%) for
    /// the current epoch. Effective commission for the next epoch
    /// lives on [`Validator::next_epoch_commission_rate`].
    pub commission_rate: u64,
}

impl StakingPool {
    pub fn new(id: ObjectID) -> Self {
        Self {
            id,
            activation_epoch: None,
            deactivation_epoch: None,
            active_stake: 0,
            pending_active_stake: 0,
            // Index starts at 1.0 (scaled). Every staker's
            // index_at_last_collect at the moment they stake equals
            // the pool's then-current cumulative_index, so the
            // ratio current/at_last_collect is 1.0 (= no compound)
            // until the next reward deposit.
            cumulative_index: F1_INDEX_SCALE,
            index_history: vec![F1_INDEX_SCALE],
            commission_rate: 0,
        }
    }

    pub fn is_inactive(&self) -> bool {
        self.deactivation_epoch.is_some()
    }

    pub fn is_preactive(&self) -> bool {
        self.activation_epoch.is_none()
    }

    /// Add stake immediately to [`active_stake`]. Used by
    /// preactive pool AddStakes (no pending semantics) and by
    /// validator self-stake at genesis. Mid-epoch active-pool stakes
    /// should use the pending bucket via [`add_pending_stake`].
    pub fn add_active_stake(&mut self, amount: u64) {
        self.active_stake = self.active_stake.saturating_add(amount);
    }

    /// Decrement [`active_stake`] for `WithdrawStake` from the active
    /// portion. The mid-epoch decrement matches Sui's
    /// `pending_pool_token_withdraw` behavior — the smaller divisor at
    /// the next fold redistributes the forfeited share to remaining
    /// stakers.
    ///
    /// Saturating: an underflow indicates corruption upstream. The
    /// executor validates `amount <= delegation.principal` before
    /// calling.
    pub fn remove_active_stake(&mut self, amount: u64) {
        self.active_stake = self.active_stake.saturating_sub(amount);
    }

    /// Bump the pending-active bucket. Called by mid-epoch `AddStake`
    /// on an active pool. Promoted into [`active_stake`] at the next
    /// epoch boundary in [`Self::process_pending_active_stake`].
    pub fn add_pending_stake(&mut self, amount: u64) {
        self.pending_active_stake = self.pending_active_stake.saturating_add(amount);
    }

    /// Decrement the pending-active bucket. Called when withdrawal
    /// drains the staker's pending portion (a same-epoch deposit
    /// reversal — that stake never earned anything).
    pub fn remove_pending_stake(&mut self, amount: u64) {
        self.pending_active_stake = self.pending_active_stake.saturating_sub(amount);
    }

    // ---------------------------------------------------------------
    // F1 helpers
    // ---------------------------------------------------------------

    /// `cumulative_index` at the start of `epoch`, where `epoch` is
    /// the absolute epoch number. Out-of-range queries clamp to the
    /// nearest available snapshot.
    ///
    /// `index_history` is indexed relative to `activation_epoch`:
    /// `index_history[0]` is the index at the moment of activation
    /// (always [`F1_INDEX_SCALE`]), `index_history[k]` is the index
    /// at the start of `activation_epoch + k` (i.e. just after the
    /// previous epoch's reward fold). For preactive pools, only
    /// `index_history[0] == F1_INDEX_SCALE` is meaningful.
    pub fn index_at_epoch_start(&self, epoch: u64) -> u128 {
        let activation = match self.activation_epoch {
            Some(a) => a,
            None => return F1_INDEX_SCALE,
        };
        if epoch <= activation {
            return self.index_history.first().copied().unwrap_or(F1_INDEX_SCALE);
        }
        let offset = (epoch - activation) as usize;
        if let Some(&v) = self.index_history.get(offset) {
            v
        } else {
            // Past end of history — clamp to latest snapshot.
            self.index_history.last().copied().unwrap_or(F1_INDEX_SCALE)
        }
    }

    /// Compute the unclaimed compound reward for a delegator with
    /// given `principal` whose baseline was `index_at_last_collect`.
    /// `auto_settle` adds this to the principal (compound).
    ///
    /// Multiplicative form: `compounded_value = principal *
    /// current_index / index_at_last_collect`, returned amount is
    /// `compounded_value - principal`. Treats a zero-baseline as a
    /// no-compound (defensive — every Delegation row is created with
    /// `index_at_last_collect == F1_INDEX_SCALE` at minimum).
    pub fn pending_compound(&self, principal: u64, index_at_last_collect: u128) -> u64 {
        if principal == 0
            || index_at_last_collect == 0
            || self.cumulative_index <= index_at_last_collect
        {
            return 0;
        }
        let compounded = (principal as u128)
            .saturating_mul(self.cumulative_index)
            / index_at_last_collect;
        compounded.saturating_sub(principal as u128) as u64
    }

    // ---------------------------------------------------------------
    // Epoch boundary operations
    // ---------------------------------------------------------------

    /// Deposit a post-commission reward into the pool. Called at
    /// every epoch boundary by
    /// [`Validator::deposit_staker_rewards`].
    ///
    /// Mirrors Sui's `deposit_rewards`: the reward grows
    /// [`active_stake`] (so future rewards' divisor includes it),
    /// and the multiplicative `cumulative_index` scales by the same
    /// ratio `(active_stake + amount) / active_stake` — existing
    /// stakers' compound share grows by exactly that ratio.
    ///
    /// Computed as
    /// `cumulative_index += cumulative_index × amount / active_stake`,
    /// algebraically equivalent to
    /// `cumulative_index *= (active_stake + amount) / active_stake`
    /// but staying within u128 range for realistic stake/index sizes.
    ///
    /// If [`active_stake`] is zero (preactive or fully-withdrawn
    /// pool), the reward still grows [`active_stake`] but the
    /// cumulative index does not advance — there's no current stake
    /// to compound for. The first new staker effectively absorbs the
    /// prior reward, matching Sui's preactive 1:1 absorption behavior.
    pub fn deposit_staker_rewards(&mut self, amount: u64) {
        if amount == 0 {
            return;
        }
        if self.active_stake > 0 {
            let increment = self
                .cumulative_index
                .saturating_mul(amount as u128)
                / (self.active_stake as u128);
            self.cumulative_index = self.cumulative_index.saturating_add(increment);
        }
        self.active_stake = self.active_stake.saturating_add(amount);
    }

    /// Promote pending stake into active. Called at epoch boundary
    /// AFTER [`deposit_staker_rewards`] — so the just-folded reward
    /// goes to existing stakers only, not to mid-epoch joiners.
    pub fn process_pending_active_stake(&mut self) {
        self.active_stake = self.active_stake.saturating_add(self.pending_active_stake);
        self.pending_active_stake = 0;
    }

    /// Snapshot the current `cumulative_index` to history. Called at
    /// the end of every epoch boundary processing — establishes the
    /// baseline for delegation rows whose pending matured this epoch.
    pub fn snapshot_index_history(&mut self) {
        self.index_history.push(self.cumulative_index);
    }

    /// Full epoch boundary processing for this pool. Order matters:
    /// 1. Deposit rewards (advances `cumulative_index`, grows
    ///    `active_stake` by reward — pre-promotion divisor)
    /// 2. Promote pending → active (for next epoch)
    /// 3. Snapshot the new `cumulative_index` (so rows with pending
    ///    promoted in this boundary use this snapshot as baseline)
    pub fn advance_epoch(&mut self, validator_reward: u64) {
        self.deposit_staker_rewards(validator_reward);
        self.process_pending_active_stake();
        self.snapshot_index_history();
    }
}

/// Per-(pool_id, staker) delegation row. Auto-compound model:
/// [`principal`] grows on every [`auto_settle`] call as the staker's
/// share of fold rewards becomes part of their stake.
///
/// Cash is realized only via `WithdrawStake`, which decrements
/// `principal` and credits the staker's SOMA balance.
///
/// `Default` returns an empty row with `index_at_last_collect == 0`
/// — the auto-settle path treats a zero baseline as "nothing to
/// compound". A first-touch executor must overwrite
/// `index_at_last_collect` with the pool's current
/// `cumulative_index` before adding any active principal so that
/// future settles read the correct ratio.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub struct Delegation {
    /// Active principal — the staker's stake currently earning. Grows
    /// on `auto_settle` (compound). Decreases on `WithdrawStake`.
    pub principal: u64,

    /// Snapshot of [`StakingPool::cumulative_index`] at the staker's
    /// last `auto_settle` call. The accrued share since then is
    /// `principal × (pool.cumulative_index - index_at_last_collect)
    /// / F1_INDEX_SCALE`.
    pub index_at_last_collect: u128,

    /// Stake added by this staker during the current epoch. Does NOT
    /// earn current-epoch rewards. Promoted into [`principal`] at the
    /// staker's next interaction *after* the next epoch boundary
    /// (lazy, via [`auto_settle`]).
    pub pending_principal: u64,

    /// The epoch in which [`pending_principal`] was last bumped.
    /// Sentinel value 0 indicates no pending (always paired with
    /// `pending_principal == 0`).
    ///
    /// On the staker's next interaction, if
    /// `pool.activation_epoch + index_history.len() > pending_added_at_epoch + 1`,
    /// the pending has matured and is promoted. Its accrued share is
    /// computed from `pool.index_at_epoch_start(pending_added_at_epoch + 1)`
    /// (the index right when it became active).
    pub pending_added_at_epoch: u64,
}

impl Delegation {
    pub fn new(principal: u64, index_at_last_collect: u128) -> Self {
        Self {
            principal,
            index_at_last_collect,
            pending_principal: 0,
            pending_added_at_epoch: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.principal == 0 && self.pending_principal == 0
    }

    /// Total stake position (active + pending). Useful for UX and
    /// withdrawal validation.
    pub fn total(&self) -> u64 {
        self.principal.saturating_add(self.pending_principal)
    }
}

/// Auto-settle: compound any accrued reward into
/// `delegation.principal`, promote any matured `pending_principal`
/// into `principal` along with its accrued share. After this call,
/// `delegation.index_at_last_collect == pool.cumulative_index` and
/// `pending_principal == 0` if matured.
///
/// Idempotent: calling on a freshly-settled row is a no-op.
///
/// Called at the start of every executor that touches a delegation
/// row (AddStake, WithdrawStake) before the row is mutated by the
/// operation itself.
///
/// Note: this function does NOT mutate `pool.active_stake` — the
/// auto-compounded rewards are *already* part of `active_stake`
/// (they were added by `deposit_staker_rewards` at boundary). This
/// function just lets the staker's row catch up to that bookkeeping.
pub fn auto_settle(delegation: &mut Delegation, pool: &StakingPool, current_epoch: u64) {
    let current_index = pool.cumulative_index;

    // (a) Compound active-principal accrual.
    if delegation.principal > 0
        && delegation.index_at_last_collect > 0
        && current_index > delegation.index_at_last_collect
    {
        let compounded = (delegation.principal as u128)
            .saturating_mul(current_index)
            / delegation.index_at_last_collect;
        delegation.principal = compounded as u64;
    }

    // (b) Promote pending if matured.
    if delegation.pending_principal > 0 && current_epoch > delegation.pending_added_at_epoch {
        // Baseline for the promoted principal is the index at the
        // start of `pending_added_at_epoch + 1` — right after that
        // epoch's reward fold, i.e. the moment pending became active.
        let baseline = pool.index_at_epoch_start(delegation.pending_added_at_epoch + 1);
        let promoted = if baseline > 0 && current_index > baseline {
            ((delegation.pending_principal as u128)
                .saturating_mul(current_index)
                / baseline) as u64
        } else {
            delegation.pending_principal
        };

        delegation.principal = delegation.principal.saturating_add(promoted);
        delegation.pending_principal = 0;
        delegation.pending_added_at_epoch = 0;
    }

    delegation.index_at_last_collect = current_index;
}
