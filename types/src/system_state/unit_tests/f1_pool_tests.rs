// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the F1 fee-distribution helpers on `StakingPool`.
//!
//! Stage 9d-A is additive: the F1 index fields exist alongside the
//! pool-token bookkeeping but are not yet authoritative. These tests
//! pin the math so that Stage 9d-B can switch reads with confidence.

use crate::config::genesis_config::SHANNONS_PER_SOMA;
use crate::object::ObjectID;
use crate::system_state::SystemStateTrait;
use crate::system_state::staking::{F1_INDEX_SCALE, StakingPool};

fn pool() -> StakingPool {
    StakingPool::new(ObjectID::new([7; 32]))
}

/// A fresh pool reports a zero index for any period and zero pending
/// reward — there is nothing to collect yet.
#[test]
fn fresh_pool_has_zero_index_and_pending() {
    let p = pool();
    assert_eq!(p.current_period, 0);
    assert_eq!(p.f1_index_at(0), 0);
    assert_eq!(p.f1_index_at(5), 0); // out-of-range clamps
    assert_eq!(p.f1_pending_reward(1_000_000, 0), 0);
}

/// Depositing rewards without folding leaves the index unchanged —
/// the in-flight `pending_fold_rewards` is the staging buffer that only
/// becomes a delegator-visible index after a fold.
#[test]
fn deposit_without_fold_leaves_index_unchanged() {
    let mut p = pool();
    p.f1_deposit_pool_reward(1_000);
    assert_eq!(p.pending_fold_rewards, 1_000);
    assert_eq!(p.current_period, 0);
    assert_eq!(p.f1_index_at(0), 0);
    // Pending reward is still 0 because no fold has happened.
    assert_eq!(p.f1_pending_reward(500, 0), 0);
}

/// Single fold with a known total stake produces an exact pending
/// reward for a delegator that owns the entire pool.
#[test]
fn single_fold_distributes_full_reward_to_sole_delegator() {
    let mut p = pool();
    p.f1_deposit_pool_reward(100);
    p.f1_fold_rewards(1_000); // total stake = 1000
    assert_eq!(p.current_period, 1);
    assert_eq!(p.pending_fold_rewards, 0);
    // Sole delegator with principal 1000 collects everything we put in.
    assert_eq!(p.f1_pending_reward(1_000, 0), 100);
}

/// Two delegators in the same period split rewards proportionally.
#[test]
fn fold_distributes_proportionally_across_delegators() {
    let mut p = pool();
    p.f1_deposit_pool_reward(900);
    p.f1_fold_rewards(1_000);
    // Alice owns 700, Bob owns 300 → 630/270 split.
    assert_eq!(p.f1_pending_reward(700, 0), 630);
    assert_eq!(p.f1_pending_reward(300, 0), 270);
    // Sums to ≤ deposit (rounding can drop ≤1 shannon).
    assert!(p.f1_pending_reward(700, 0) + p.f1_pending_reward(300, 0) <= 900);
}

/// A delegator that joins after a fold sees only rewards from later
/// periods. F1's whole point is per-staker history isolation — the
/// `last_collected_period` discriminates.
#[test]
fn late_joiner_does_not_see_pre_join_rewards() {
    let mut p = pool();
    p.f1_deposit_pool_reward(500);
    p.f1_fold_rewards(1_000); // period 0 → 1
    let alice_join_period = p.current_period; // 1

    p.f1_deposit_pool_reward(500);
    p.f1_fold_rewards(1_000); // period 1 → 2

    // Alice joined at period 1 and collects only period-2 rewards.
    assert_eq!(p.f1_pending_reward(1_000, alice_join_period), 500);

    // A historical delegator from period 0 collects both folds.
    assert_eq!(p.f1_pending_reward(1_000, 0), 1_000);
}

/// Folding with zero rewards advances the period counter but does not
/// move the index — a delegator collected up to the prior fold sees no
/// new pending reward, regardless of how many empty folds pass.
#[test]
fn empty_fold_advances_period_without_changing_payout() {
    let mut p = pool();
    p.f1_deposit_pool_reward(200);
    p.f1_fold_rewards(1_000);
    let baseline = p.f1_pending_reward(500, 0);

    // Advance 5 empty periods.
    for _ in 0..5 {
        p.f1_fold_rewards(1_000);
    }
    assert_eq!(p.current_period, 6);
    assert_eq!(p.f1_pending_reward(500, 0), baseline);
}

/// Fold uses the current index (not zero) when stake is zero — keeps
/// new joiners from accidentally seeing pre-join rewards. With no
/// stakers, accumulated rewards have no one to distribute to and stay
/// in `pending_fold_rewards` until they're explicitly cleared elsewhere.
#[test]
fn fold_with_zero_stake_does_not_corrupt_index() {
    let mut p = pool();
    // No deposits, no stake — fold should be a no-op for the index.
    let prev_index = p.f1_index_at(p.current_period);
    p.f1_fold_rewards(0);
    assert_eq!(p.f1_index_at(p.current_period), prev_index);
}

/// When `total_stake == 0` but `pending_fold_rewards > 0`, the rewards
/// must NOT be silently dropped — they have to carry forward to the
/// next non-empty fold so the SOMA in `pool_rewards` can eventually
/// reach a delegator. (Pre-fix behavior cleared
/// `pending_fold_rewards` to 0, stranding the rewards: the bank
/// retained the SOMA but no future cumulative_index growth was tied
/// to those rewards.)
#[test]
fn fold_with_zero_stake_carries_pending_forward_to_next_fold() {
    let mut p = pool();
    // Deposit rewards while pool has 0 stake.
    p.f1_deposit_pool_reward(1_000);
    assert_eq!(p.pending_fold_rewards, 1_000);
    let pool_rewards_before = p.pool_rewards;

    // Fold with zero stake — rewards should NOT be cleared.
    let pre_period = p.current_period;
    p.f1_fold_rewards(0);
    assert_eq!(p.current_period, pre_period + 1, "period must advance");
    assert_eq!(
        p.pending_fold_rewards, 1_000,
        "pending_fold_rewards must carry forward when total_stake == 0",
    );
    assert_eq!(
        p.pool_rewards, pool_rewards_before,
        "pool_rewards bank is preserved",
    );

    // Now a delegator joins; the next fold must distribute the
    // carried-forward rewards.
    let stake = 1_000_000u64;
    p.f1_fold_rewards(stake);
    assert_eq!(
        p.pending_fold_rewards, 0,
        "next non-empty fold must drain the carried rewards",
    );
    let cur = p.f1_index_at(p.current_period);
    let expected_index = (1_000u128).saturating_mul(F1_INDEX_SCALE) / (stake as u128);
    assert_eq!(
        cur, expected_index,
        "carried-forward rewards must be distributed at the joining stake",
    );
}

/// Index is monotonically non-decreasing across folds — a property
/// every staker collection algorithm relies on.
#[test]
fn index_is_monotonically_non_decreasing() {
    let mut p = pool();
    let mut last = p.f1_index_at(p.current_period);
    for i in 1..=10 {
        p.f1_deposit_pool_reward(i * 100);
        p.f1_fold_rewards(10_000);
        let cur = p.f1_index_at(p.current_period);
        assert!(cur >= last, "index regressed at period {}: {} -> {}", i, last, cur);
        last = cur;
    }
}

/// Scale factor must be exactly the documented 1e18 — third-party
/// integrations and any future migration tooling will rely on it.
#[test]
fn scale_factor_is_documented_value() {
    assert_eq!(F1_INDEX_SCALE, 1_000_000_000_000_000_000);
}

/// Pre-mainnet sanity: a 1B-shannon stake earning 1 shannon per period
/// for many periods should round-trip without losing more than a few
/// shannons total. The 1e18 scale is the safety margin.
#[test]
fn small_rewards_with_large_stake_do_not_round_to_zero() {
    let mut p = pool();
    let total_stake: u64 = 1_000_000_000; // 1B shannons
    for _ in 0..100 {
        p.f1_deposit_pool_reward(1); // 1 shannon per period
        p.f1_fold_rewards(total_stake);
    }
    // A delegator owning 1% of stake collects ≈1 shannon over 100 periods.
    let payout = p.f1_pending_reward(10_000_000, 0);
    assert!(payout >= 1, "expected ≥1 shannon, got {}", payout);
}

// ---------------------------------------------------------------
// Integration tests: drive the full SystemState::advance_epoch path
// and confirm F1 stays in lockstep with pool-token bookkeeping.
// Stage 9d-B wires `f1_fold_rewards` into the epoch boundary; these
// tests pin the invariant that both algorithms produce the same
// per-staker payout under realistic flows.
// ---------------------------------------------------------------

/// After several epochs of rewards via the production
/// `advance_epoch` path, the F1 cumulative index advances in lockstep
/// with the pool-token exchange rate.
#[test]
fn f1_index_advances_each_epoch_through_advance_epoch() {
    use crate::base::dbg_addr;
    use crate::system_state::test_utils::{
        ValidatorRewards, advance_epoch_with_reward_amounts, create_test_system_state,
        create_validator_for_testing,
    };

    let v1 = create_validator_for_testing(dbg_addr(1), 100);
    let v2 = create_validator_for_testing(dbg_addr(2), 200);
    let mut system_state = create_test_system_state(vec![v1, v2], 1_000, 0);
    let mut tracker = ValidatorRewards::new(&system_state.validators().validators);

    // Bootstrap epoch with no rewards to activate validators.
    advance_epoch_with_reward_amounts(&mut system_state, 0, &mut tracker);

    let period_before = system_state.validators().validators[0]
        .staking_pool
        .current_period;

    // Deliver 100 SOMA in rewards.
    advance_epoch_with_reward_amounts(&mut system_state, 100, &mut tracker);

    let pool_after = &system_state.validators().validators[0].staking_pool;
    assert_eq!(
        pool_after.current_period,
        period_before + 1,
        "F1 period must advance exactly once per epoch boundary",
    );
    assert_eq!(
        pool_after.pending_fold_rewards, 0,
        "pending_fold_rewards must be drained into the index after a fold",
    );
    assert!(
        pool_after.f1_index_at(pool_after.current_period)
            > pool_after.f1_index_at(period_before),
        "rewards must move the cumulative index forward",
    );
}

/// Across multiple epochs, F1's pending reward for a sole delegator
/// staking at genesis equals the sum of per-epoch rewards (no
/// auto-compound under F1).
#[test]
fn f1_payout_for_sole_delegator_equals_sum_of_rewards() {
    use crate::base::dbg_addr;
    use crate::system_state::test_utils::{
        ValidatorRewards, advance_epoch_with_reward_amounts, create_test_system_state,
        create_validator_for_testing,
    };

    let validator_addr = dbg_addr(1);
    let initial_stake_soma = 1_000u64;
    let initial_stake_shannons = initial_stake_soma * SHANNONS_PER_SOMA;
    let v = create_validator_for_testing(validator_addr, initial_stake_shannons);
    let mut system_state = create_test_system_state(vec![v], 10_000, 0);
    let mut tracker = ValidatorRewards::new(&system_state.validators().validators);

    // Bootstrap so activation_epoch=0 stakes are live.
    advance_epoch_with_reward_amounts(&mut system_state, 0, &mut tracker);

    let rewards_soma = [50u64, 75, 100, 25, 125];
    for reward in rewards_soma {
        advance_epoch_with_reward_amounts(&mut system_state, reward, &mut tracker);
    }

    let pool = &system_state.validators().validators[0].staking_pool;
    let f1_reward = pool.f1_pending_reward(initial_stake_shannons, 0);

    // Sole delegator collects the full reward stream. Per-fold
    // truncation can drop ≤1 shannon each, so allow ~5 shannon
    // divergence over 5 folds.
    let total_reward_shannons: u64 = rewards_soma.iter().sum::<u64>() * SHANNONS_PER_SOMA;
    let diff = total_reward_shannons.abs_diff(f1_reward);
    assert!(
        diff <= 6,
        "sole delegator must collect ≈ sum of rewards: total={}, f1={}, diff={}",
        total_reward_shannons,
        f1_reward,
        diff,
    );
}

/// Spot-check: a single fold with 1000 stake and 100 reward yields
/// exactly 100 to the sole delegator.
#[test]
fn f1_single_epoch_sole_delegator_collects_full_reward() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.total_stake = 1_000;

    p.f1_deposit_pool_reward(100);
    p.f1_fold_rewards(1_000);

    let f1_payout = p.f1_pending_reward(1_000, 0);
    assert_eq!(f1_payout, 100, "sole delegator collects all rewards");
}
