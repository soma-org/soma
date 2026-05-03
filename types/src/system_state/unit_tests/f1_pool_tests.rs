// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the F1 fee-distribution helpers on `StakingPool`.
//!
//! Stage 9d-A is additive: the F1 index fields exist alongside the
//! pool-token bookkeeping but are not yet authoritative. These tests
//! pin the math so that Stage 9d-B can switch reads with confidence.

use crate::object::ObjectID;
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
/// the in-flight `current_rewards` is the staging buffer that only
/// becomes a delegator-visible index after a fold.
#[test]
fn deposit_without_fold_leaves_index_unchanged() {
    let mut p = pool();
    p.f1_deposit_pool_reward(1_000);
    assert_eq!(p.current_rewards, 1_000);
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
    assert_eq!(p.current_rewards, 0);
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
/// in `current_rewards` until they're explicitly cleared elsewhere.
#[test]
fn fold_with_zero_stake_does_not_corrupt_index() {
    let mut p = pool();
    // No deposits, no stake — fold should be a no-op for the index.
    let prev_index = p.f1_index_at(p.current_period);
    p.f1_fold_rewards(0);
    assert_eq!(p.f1_index_at(p.current_period), prev_index);
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

/// F1 path agrees with the pool-token path for the simple case of a
/// sole delegator over a single epoch of rewards. Both algorithms must
/// say "1000 principal + 100 reward = 1100 total" — Stage 9d-B's
/// switch-over depends on this equivalence.
#[test]
fn f1_matches_pool_token_for_sole_delegator_one_epoch() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.soma_balance = 1_000;
    p.pool_token_balance = 1_000;
    p.exchange_rates.insert(
        0,
        crate::system_state::staking::PoolTokenExchangeRate {
            soma_amount: 1_000,
            pool_token_amount: 1_000,
        },
    );

    // Simulate one epoch boundary worth of rewards via the same
    // sequence `Validator::deposit_staker_rewards` runs in production:
    // pool-token side deposits then advances exchange rate; F1 side
    // deposits then folds at the pre-reward stake.
    p.deposit_rewards(100);
    p.f1_deposit_pool_reward(100);
    p.f1_fold_rewards(1_000); // pre-reward stake as denominator
    p.update_exchange_rate(1); // pool-token snapshots end-of-epoch rate

    let pool_token_payout = p.calculate_rewards(1_000, 0, 1) - 1_000;
    let f1_payout = p.f1_pending_reward(1_000, 0);
    assert_eq!(pool_token_payout, f1_payout, "F1 must match pool-token math");
    assert_eq!(f1_payout, 100, "sole delegator collects all rewards");
}
