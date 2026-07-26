// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the auto-compound F1 [`StakingPool`].
//!
//! Mirrors the structural intent of Sui's `staking_pool_tests.move`:
//! pool lifecycle (preactive → active → inactive), reward deposit
//! advancing the cumulative index proportionally to existing stake,
//! pending bucket promotion at the next boundary, no-underflow on
//! mid-epoch withdrawal drain, multi-staker proportional rewards.
//!
//! `auto_settle` math is exercised by [`auto_compound_settle_tests`]
//! below and indirectly by every multi-epoch test here.

use crate::object::ObjectID;
use crate::system_state::staking::{Delegation, F1_INDEX_SCALE, StakingPool, auto_settle};

fn pool() -> StakingPool {
    StakingPool::new(ObjectID::new([7; 32]))
}

/// Activate at epoch 0 and step the boundary to epoch 1 with no
/// rewards. Mirrors Sui's `distribute_rewards_and_advance_epoch` —
/// every multi-epoch test routes through this so the pre-condition
/// for next-epoch reads is consistent.
fn distribute_and_advance(p: &mut StakingPool, reward: u64) {
    p.advance_epoch(reward);
}

// =============================================================================
// Pool lifecycle: preactive / active / inactive
// =============================================================================

/// A fresh pool is preactive — no activation epoch, zero stake,
/// index seeded at `F1_INDEX_SCALE` (= 1.0 exchange rate). Matches
/// Sui's `staking_pool::new` invariants.
#[test]
fn fresh_pool_is_preactive() {
    let p = pool();
    assert!(p.is_preactive());
    assert!(!p.is_inactive());
    assert_eq!(p.active_stake, 0);
    assert_eq!(p.pending_active_stake, 0);
    assert_eq!(p.cumulative_index, F1_INDEX_SCALE);
    assert_eq!(p.index_history, vec![F1_INDEX_SCALE]);
}

/// Activation flips the preactive flag without touching stake or
/// index. The index_history preserves its genesis snapshot.
#[test]
fn activate_sets_activation_epoch() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    assert!(!p.is_preactive());
    assert!(!p.is_inactive());
    assert_eq!(p.cumulative_index, F1_INDEX_SCALE);
}

/// Deactivation flips the inactive flag, again without touching
/// stake — pending withdrawals continue to settle against the frozen
/// snapshot of `cumulative_index`.
#[test]
fn deactivate_sets_deactivation_epoch() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.deactivation_epoch = Some(5);
    assert!(p.is_inactive());
}

// =============================================================================
// deposit_staker_rewards
// =============================================================================

/// Depositing rewards onto an empty pool grows `active_stake` but
/// does not advance the cumulative index — there's nothing to
/// compound for. Matches Sui's preactive 1:1 absorption.
#[test]
fn deposit_with_zero_stake_grows_balance_not_index() {
    let mut p = pool();
    p.deposit_staker_rewards(1_000);
    assert_eq!(p.active_stake, 1_000);
    assert_eq!(p.cumulative_index, F1_INDEX_SCALE, "index unchanged");
}

/// Standard case: deposit `R` onto a pool with active stake `S`
/// scales `cumulative_index` by `(S + R) / S` — the multiplicative
/// exchange rate update. After the deposit, `active_stake` reflects
/// S + R so the next deposit's divisor includes the just-folded
/// reward (auto-compound).
#[test]
fn deposit_scales_index_by_growth_ratio() {
    let mut p = pool();
    p.active_stake = 1_000_000_000; // 1 SOMA
    p.deposit_staker_rewards(100_000_000); // 0.1 SOMA reward

    // Multiplicative: index *= 1.1B/1B = 1.1
    let expected = F1_INDEX_SCALE + F1_INDEX_SCALE / 10;
    assert_eq!(p.cumulative_index, expected);
    assert_eq!(p.active_stake, 1_100_000_000);
}

/// Two consecutive deposits compound multiplicatively — the second
/// deposit scales the *already-scaled* index. After two 10% deposits
/// the index reflects (1.1 × 1.0909...) ≈ 1.2.
#[test]
fn consecutive_deposits_compound_multiplicatively() {
    let mut p = pool();
    p.active_stake = 1_000_000_000;
    p.deposit_staker_rewards(100_000_000); // active_stake → 1.1 SOMA
    let after_first = p.cumulative_index;

    p.deposit_staker_rewards(100_000_000); // index *= 1.2/1.1
    // increment = old_index × amount / active_pre = 1.1×SCALE × 100M / 1.1B
    let expected_increment = after_first * 100_000_000u128 / 1_100_000_000u128;
    let expected_total = after_first + expected_increment;
    assert_eq!(p.cumulative_index, expected_total);
    assert_eq!(p.active_stake, 1_200_000_000);

    // Sanity: total ratio = active_post / active_initial = 1.2
    // (within 1 part per 1e18 of integer rounding).
    let expected_full_ratio = F1_INDEX_SCALE * 12 / 10;
    let diff = expected_total.abs_diff(expected_full_ratio);
    assert!(
        diff <= 1,
        "compounded ratio matches 1.2× within rounding, got {} vs {}",
        expected_total,
        expected_full_ratio
    );
}

/// Zero-reward deposit is a no-op — index doesn't move, stake
/// doesn't grow. Matches Sui's `deposit_rewards` early-return on
/// empty balance.
#[test]
fn deposit_zero_is_noop() {
    let mut p = pool();
    p.active_stake = 100;
    p.deposit_staker_rewards(0);
    assert_eq!(p.active_stake, 100);
    assert_eq!(p.cumulative_index, F1_INDEX_SCALE);
}

// =============================================================================
// pending_active_stake bucket
// =============================================================================

/// Mid-epoch additions land in the pending bucket. They do NOT count
/// toward `active_stake` (the fold divisor), so the next reward
/// deposit goes only to existing stakers.
#[test]
fn pending_addition_does_not_dilute_current_epoch_rewards() {
    let mut p = pool();
    p.active_stake = 1_000_000_000;

    // Mid-epoch staker joins.
    p.add_pending_stake(500_000_000);
    assert_eq!(p.active_stake, 1_000_000_000);
    assert_eq!(p.pending_active_stake, 500_000_000);

    // Boundary deposit. The 0.5 SOMA pending stake must NOT receive
    // any of this reward — divisor is the existing 1 SOMA only.
    p.deposit_staker_rewards(100_000_000);
    // Index scales by 1.1 (10% growth).
    let expected = F1_INDEX_SCALE + F1_INDEX_SCALE / 10;
    assert_eq!(p.cumulative_index, expected, "index advance only against existing 1 SOMA");
    assert_eq!(p.active_stake, 1_100_000_000);
    assert_eq!(p.pending_active_stake, 500_000_000, "pending untouched by deposit");
}

/// `process_pending_active_stake` promotes the bucket atomically.
/// Called at the epoch boundary AFTER `deposit_staker_rewards`, so
/// pending stake earns rewards starting from the *next* epoch.
#[test]
fn pending_promotes_after_deposit() {
    let mut p = pool();
    p.active_stake = 1_000_000_000;
    p.add_pending_stake(500_000_000);

    p.deposit_staker_rewards(100_000_000); // existing-stake-only
    p.process_pending_active_stake();

    assert_eq!(p.active_stake, 1_600_000_000, "promoted: 1.1 SOMA + 0.5 SOMA pending");
    assert_eq!(p.pending_active_stake, 0);
}

/// `advance_epoch` is the boundary helper — deposit, promote,
/// snapshot. The snapshot establishes the baseline for delegation
/// rows whose pending matured in this boundary.
#[test]
fn advance_epoch_snapshots_index_history() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.active_stake = 1_000_000_000;

    p.advance_epoch(100_000_000);
    assert_eq!(p.index_history.len(), 2, "genesis snapshot + epoch 0 snapshot");
    assert_eq!(p.index_history[1], p.cumulative_index, "snapshot equals post-fold index");

    p.advance_epoch(100_000_000);
    assert_eq!(p.index_history.len(), 3);
    assert_eq!(p.index_history[2], p.cumulative_index);
}

// =============================================================================
// Mid-epoch withdrawal redistribution (Sui's pending_pool_token_withdraw)
// =============================================================================

/// A mid-epoch withdrawal decrements `active_stake` immediately. At
/// the next boundary, the smaller divisor redistributes the
/// withdrawer's would-have-been-share to remaining stakers — the F1
/// equivalent of Sui's pool-token redistribution.
#[test]
fn mid_epoch_withdrawal_redistributes_share_at_next_fold() {
    let mut p = pool();
    p.active_stake = 1_000_000_000;

    // Alice withdraws 500M mid-epoch, leaving 500M.
    p.remove_active_stake(500_000_000);
    assert_eq!(p.active_stake, 500_000_000);

    // Boundary deposit of 100M against the smaller divisor.
    p.deposit_staker_rewards(100_000_000);
    // Index scales by 600M/500M = 1.2 (20% growth) — bigger jump
    // than 1.1 because the divisor was the post-withdrawal 500M.
    let expected = F1_INDEX_SCALE + F1_INDEX_SCALE / 5;
    assert_eq!(p.cumulative_index, expected);
}

/// Drain the pool fully mid-epoch. The next deposit must not panic
/// even though `active_stake` is zero — the deposit grows the pool
/// without advancing the index, mirroring `deposit_with_zero_stake`.
#[test]
fn full_drain_then_deposit_no_underflow() {
    let mut p = pool();
    p.active_stake = 1_000_000_000;
    p.remove_active_stake(1_000_000_000);
    assert_eq!(p.active_stake, 0);

    p.deposit_staker_rewards(100_000_000);
    assert_eq!(p.active_stake, 100_000_000);
    assert_eq!(p.cumulative_index, F1_INDEX_SCALE, "no index advance — no stake to compound for",);
}

/// `remove_active_stake` saturates rather than panicking on
/// underflow. The executor enforces `amount <= principal` before
/// calling — saturation here is a defense-in-depth.
#[test]
fn remove_more_than_present_saturates() {
    let mut p = pool();
    p.active_stake = 100;
    p.remove_active_stake(500);
    assert_eq!(p.active_stake, 0);
}

// =============================================================================
// index_at_epoch_start: relative-to-activation lookups
// =============================================================================

/// `index_at_epoch_start(activation)` returns the genesis snapshot
/// (= `F1_INDEX_SCALE`); `index_at_epoch_start(activation + k)`
/// returns the post-fold snapshot for epoch `activation + k - 1`.
#[test]
fn index_at_epoch_start_walks_history() {
    let mut p = pool();
    p.activation_epoch = Some(10);
    p.active_stake = 1_000_000_000;

    p.advance_epoch(100_000_000); // epoch 10 closes; index_history[1] = post-fold
    let after_e10 = p.cumulative_index;

    p.advance_epoch(100_000_000); // epoch 11 closes; index_history[2] = post-fold
    let after_e11 = p.cumulative_index;

    assert_eq!(p.index_at_epoch_start(10), F1_INDEX_SCALE, "activation epoch starts at 1.0");
    assert_eq!(p.index_at_epoch_start(11), after_e10, "start of e11 = post e10");
    assert_eq!(p.index_at_epoch_start(12), after_e11, "start of e12 = post e11");
}

/// Out-of-range queries clamp to the latest snapshot — the current
/// `cumulative_index`. Called by `auto_settle` for fast-forwarded
/// reads.
#[test]
fn index_at_epoch_start_out_of_range_clamps() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.active_stake = 1_000_000_000;
    p.advance_epoch(100_000_000);

    assert_eq!(p.index_at_epoch_start(999), p.cumulative_index);
}

/// Preactive pool: history is meaningless. Any query returns the
/// genesis snapshot value (`F1_INDEX_SCALE`).
#[test]
fn index_at_epoch_start_preactive_is_unit() {
    let p = pool();
    assert_eq!(p.index_at_epoch_start(0), F1_INDEX_SCALE);
    assert_eq!(p.index_at_epoch_start(7), F1_INDEX_SCALE);
}

// =============================================================================
// Multi-staker proportional rewards (Sui parity)
// =============================================================================

/// Two stakers, equal principal, equal share of a single deposit.
/// Mirrors Sui's `convert_to_fungible_staked_sui_happy` — same
/// exchange rate for stakers who entered at the same epoch.
#[test]
fn equal_principal_equal_compound() {
    let mut p = pool();
    p.activation_epoch = Some(0);

    // Both seeded at the genesis exchange rate (1.0).
    let mut alice = Delegation::new(1_000_000_000, F1_INDEX_SCALE);
    let mut bob = Delegation::new(1_000_000_000, F1_INDEX_SCALE);
    p.active_stake = 2_000_000_000;

    p.advance_epoch(200_000_000); // 0.2 SOMA shared

    auto_settle(&mut alice, &p, 1);
    auto_settle(&mut bob, &p, 1);
    assert_eq!(alice.principal, 1_100_000_000, "alice gets half of 0.2 SOMA = 0.1 SOMA");
    assert_eq!(bob.principal, 1_100_000_000, "bob gets the other half");
}

/// Unequal principal → proportional reward. Matches Sui's
/// pool-token math where pool tokens mint at the current exchange
/// rate.
#[test]
fn proportional_reward_split() {
    let mut p = pool();
    p.activation_epoch = Some(0);

    let mut alice = Delegation::new(3_000_000_000, F1_INDEX_SCALE); // 75%
    let mut bob = Delegation::new(1_000_000_000, F1_INDEX_SCALE); //   25%
    p.active_stake = 4_000_000_000;

    p.advance_epoch(400_000_000); // 0.4 SOMA pool reward

    auto_settle(&mut alice, &p, 1);
    auto_settle(&mut bob, &p, 1);
    assert_eq!(alice.principal, 3_300_000_000, "alice 75% → 0.3 SOMA compound");
    assert_eq!(bob.principal, 1_100_000_000, "bob 25% → 0.1 SOMA compound");
    assert_eq!(
        alice.principal + bob.principal,
        p.active_stake,
        "after both settle, principals equal active_stake",
    );
}

/// A staker who joined mid-epoch (pending bucket) does NOT receive
/// the boundary deposit. Sui parity: pool tokens mint *after* the
/// reward distribution.
#[test]
fn mid_epoch_staker_skips_current_epoch_reward() {
    let mut p = pool();
    p.activation_epoch = Some(0);

    let mut alice = Delegation::new(1_000_000_000, F1_INDEX_SCALE);
    p.active_stake = 1_000_000_000;

    // Carol joins mid-epoch.
    let mut carol = Delegation {
        principal: 0,
        index_at_last_collect: p.cumulative_index,
        pending_principal: 1_000_000_000,
        pending_added_at_epoch: 0,
    };
    p.add_pending_stake(1_000_000_000);

    p.advance_epoch(100_000_000); // boundary fold

    auto_settle(&mut alice, &p, 1);
    auto_settle(&mut carol, &p, 1);
    assert_eq!(alice.principal, 1_100_000_000, "alice earns the full 0.1 SOMA");
    assert_eq!(carol.principal, 1_000_000_000, "carol's pending promoted; no reward share");
    assert_eq!(carol.pending_principal, 0, "pending bucket drained on settle");
}

/// Carol's pending promotes; from the *next* epoch she earns rewards
/// proportionally with everyone else. The MULTIPLICATIVE index gives
/// the correct economic answer where alice's compound stake (1.1B)
/// keeps her share at 1.1/2.1 of subsequent rewards.
#[test]
fn matured_pending_earns_from_next_epoch() {
    let mut p = pool();
    p.activation_epoch = Some(0);

    let mut alice = Delegation::new(1_000_000_000, F1_INDEX_SCALE);
    p.active_stake = 1_000_000_000;

    let mut carol = Delegation {
        principal: 0,
        index_at_last_collect: F1_INDEX_SCALE,
        pending_principal: 1_000_000_000,
        pending_added_at_epoch: 0,
    };
    p.add_pending_stake(1_000_000_000);

    p.advance_epoch(100_000_000); // epoch 0 closes; alice earns 0.1
    p.advance_epoch(200_000_000); // epoch 1 closes; alice + carol split 0.2

    auto_settle(&mut alice, &p, 2);
    auto_settle(&mut carol, &p, 2);

    // Economics:
    // - Start of epoch 1: alice has 1.1B (compounded from epoch 0),
    //   carol has 1.0B (just-promoted). Total = 2.1B.
    // - Epoch 1 reward 0.2 SOMA splits proportionally:
    //   alice's share: 1.1/2.1 × 0.2 ≈ 104.76M
    //   carol's share: 1.0/2.1 × 0.2 ≈ 95.24M
    // - Final balances: alice ≈ 1.204_762B, carol ≈ 1.095_238B.
    // Allow ~1 shannon rounding loss per delegation.

    let alice_expected_lo = 1_204_761_900u64;
    let alice_expected_hi = 1_204_762_000u64;
    assert!(
        (alice_expected_lo..=alice_expected_hi).contains(&alice.principal),
        "alice in [{}, {}], got {}",
        alice_expected_lo,
        alice_expected_hi,
        alice.principal,
    );

    let carol_expected_lo = 1_095_238_000u64;
    let carol_expected_hi = 1_095_238_100u64;
    assert!(
        (carol_expected_lo..=carol_expected_hi).contains(&carol.principal),
        "carol in [{}, {}], got {}",
        carol_expected_lo,
        carol_expected_hi,
        carol.principal,
    );

    // Conservation: total principals match active_stake within
    // bounded rounding loss (one shannon per row).
    let unclaimed = p.active_stake.saturating_sub(alice.principal + carol.principal);
    assert!(unclaimed <= 2, "unclaimed compound bounded by rounding, got {}", unclaimed);
}

// =============================================================================
// auto_settle: idempotency, settle math
// =============================================================================

/// `auto_settle` on a freshly-settled row is a no-op.
#[test]
fn auto_settle_idempotent() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.active_stake = 1_000_000_000;
    p.advance_epoch(100_000_000);

    let mut row = Delegation::new(500_000_000, F1_INDEX_SCALE);
    auto_settle(&mut row, &p, 1);
    let after_first = row;
    auto_settle(&mut row, &p, 1);
    assert_eq!(row, after_first, "second settle is a no-op");
}

/// `auto_settle` with no accrual (delegation just refreshed) leaves
/// principal unchanged.
#[test]
fn auto_settle_no_accrual_leaves_principal() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.active_stake = 1_000_000_000;

    let mut row = Delegation::new(500_000_000, F1_INDEX_SCALE);
    auto_settle(&mut row, &p, 0);
    assert_eq!(row.principal, 500_000_000);
    assert_eq!(row.index_at_last_collect, F1_INDEX_SCALE);
}

/// `pending_compound` returns zero when there's no accrual to
/// compound. Mirrors Sui's `staked_sui_amount` returning the
/// principal for stake at the current rate.
#[test]
fn pending_compound_zero_when_index_unchanged() {
    let mut p = pool();
    p.cumulative_index = F1_INDEX_SCALE * 2; // exchange rate 2.0
    assert_eq!(p.pending_compound(1_000, F1_INDEX_SCALE * 2), 0, "same index");
    assert_eq!(p.pending_compound(0, F1_INDEX_SCALE), 0, "zero principal");
    assert_eq!(p.pending_compound(1_000, 0), 0, "zero baseline (defensive)");
}

// =============================================================================
// Rounding: Sui's staking_pool_tests has a similar regression test
// =============================================================================

/// Odd numbers should not lose more than one shannon to integer
/// division across a full settle cycle. Mirrors Sui's
/// `redeem_fungible_staked_sui_regression_rounding` in spirit.
#[test]
fn settle_rounding_loss_bounded() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.active_stake = 1_000_000_001;

    let mut alice = Delegation::new(1_000_000_001, F1_INDEX_SCALE);
    p.advance_epoch(1_000_000_000);
    auto_settle(&mut alice, &p, 1);

    // Before settle: active_stake = 2_000_000_001.
    // After settle: alice owns the entire pool (only staker).
    let unclaimed = p.active_stake - alice.principal;
    assert!(unclaimed <= 1, "rounding loss bounded by 1 shannon, got {}", unclaimed);
}

/// Long-horizon test: after many epochs of compound at a realistic
/// 5% per-epoch yield, the cumulative_index stays well within u128
/// range and a single staker who never settles still ends up with
/// the right compounded value. Mirrors Sui's stress on
/// long-running pools.
#[test]
fn long_horizon_compound_no_overflow() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.active_stake = 1_000_000_000;
    let mut alice = Delegation::new(1_000_000_000, F1_INDEX_SCALE);

    // 100 epochs of 5% yield.
    for _ in 0..100 {
        let reward = p.active_stake / 20;
        p.advance_epoch(reward);
    }

    auto_settle(&mut alice, &p, 100);
    // Expected: 1B × 1.05^100 ≈ 131.5B. Allow generous rounding
    // tolerance; main goal is no overflow, no panic.
    assert!(
        alice.principal > 130_000_000_000,
        "100 epochs of 5% yield should compound to >130 SOMA, got {}",
        alice.principal,
    );
    assert!(
        alice.principal < 135_000_000_000,
        "compound shouldn't exceed 1.05^100, got {}",
        alice.principal,
    );

    // Conservation: alice owns the whole pool (only staker).
    let diff = p.active_stake.abs_diff(alice.principal);
    assert!(diff <= 2, "rounding bounded over 100 epochs, got {}", diff);
}

// =============================================================================
// Multiplicative-correctness regression
// =============================================================================

// =============================================================================
// First-time staker on a pool that has already accrued rewards
// =============================================================================

/// A staker joining a non-fresh pool sets `index_at_last_collect`
/// to the pool's current `cumulative_index`. From that snapshot
/// onward, subsequent fold deposits produce the right
/// `current_index / index_at_last_collect` ratio. Catches a
/// regression where new stakers' baselines defaulted to 0 (or
/// `F1_INDEX_SCALE`) and silently absorbed historical rewards.
#[test]
fn fresh_staker_on_warm_pool_baselines_to_current_index() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.active_stake = 1_000_000_000;
    let mut existing = Delegation::new(1_000_000_000, F1_INDEX_SCALE);

    // Pool earns rewards before carol stakes — index advances.
    p.advance_epoch(100_000_000);
    let warm_index = p.cumulative_index;
    assert!(warm_index > F1_INDEX_SCALE);

    // Carol joins now. She must baseline to the current index, not
    // to the genesis index.
    let mut carol = Delegation::new(0, F1_INDEX_SCALE);
    carol.pending_principal = 500_000_000;
    carol.pending_added_at_epoch = 1;
    carol.index_at_last_collect = warm_index;
    p.add_pending_stake(500_000_000);

    // One more epoch with rewards — carol's pending matures, both
    // earn proportional shares.
    p.advance_epoch(160_000_000);

    auto_settle(&mut existing, &p, 2);
    auto_settle(&mut carol, &p, 2);

    // existing was settled; she compounds against the full index.
    // Started at SCALE, advanced to 1.1×SCALE after epoch 0, then
    // grew by 1660/1500 (epoch 1: deposit 160 against 1500 active).
    // Existing's compound = 1.1B × (final_index)/SCALE.
    assert!(existing.principal > 1_100_000_000, "existing keeps compounding");

    // Carol earned only epoch-1's reward (after her pending matured).
    // She did NOT receive epoch 0's 100M reward.
    assert!(carol.principal >= 500_000_000, "carol gets at least her principal back");
    assert!(carol.principal < 600_000_000, "carol does NOT receive epoch 0's reward");

    // Conservation: alice + carol ≈ active_stake.
    let unclaimed = p.active_stake.saturating_sub(existing.principal + carol.principal);
    assert!(unclaimed <= 2, "rounding bounded, got {}", unclaimed);
}

// =============================================================================
// Pool/delegation conservation across stake/withdraw cycles
// =============================================================================

/// After a sequence of stake / advance / withdraw operations, the
/// sum of every settled delegation's principal must equal the pool's
/// `active_stake` (within rounding). This is the key cross-check
/// that catches pool-aggregate vs. row drift — the bug we fixed
/// where a withdrawal's pending/active split was computed against
/// the pool aggregate instead of the staker's own row.
#[test]
fn pool_active_stake_equals_sum_of_settled_principals() {
    let mut p = pool();
    p.activation_epoch = Some(0);

    // Start with two existing stakers.
    let mut alice = Delegation::new(1_000_000_000, F1_INDEX_SCALE);
    let mut bob = Delegation::new(1_000_000_000, F1_INDEX_SCALE);
    p.active_stake = 2_000_000_000;

    // Reward epoch.
    p.advance_epoch(200_000_000);

    // Carol and Dave both add pending stake mid-epoch 1.
    p.add_pending_stake(1_000_000_000);
    let mut carol = Delegation {
        principal: 0,
        index_at_last_collect: p.cumulative_index,
        pending_principal: 600_000_000,
        pending_added_at_epoch: 1,
    };
    let mut dave = Delegation {
        principal: 0,
        index_at_last_collect: p.cumulative_index,
        pending_principal: 400_000_000,
        pending_added_at_epoch: 1,
    };

    // Reward epoch — carol & dave's pending promotes.
    p.advance_epoch(150_000_000);

    // Alice withdraws half mid-epoch 2. We model this at the row +
    // pool level (mirroring what the executor would do).
    auto_settle(&mut alice, &p, 2);
    let withdraw_amount = alice.principal / 2;
    alice.principal -= withdraw_amount;
    p.remove_active_stake(withdraw_amount);

    // Reward epoch.
    p.advance_epoch(100_000_000);

    // Settle everyone and verify conservation.
    auto_settle(&mut alice, &p, 3);
    auto_settle(&mut bob, &p, 3);
    auto_settle(&mut carol, &p, 3);
    auto_settle(&mut dave, &p, 3);

    let sum = alice.principal + bob.principal + carol.principal + dave.principal;
    let unclaimed = p.active_stake.saturating_sub(sum);
    let overclaimed = sum.saturating_sub(p.active_stake);
    assert!(
        unclaimed <= 4 && overclaimed == 0,
        "sum of principals {} must equal active_stake {} within rounding",
        sum,
        p.active_stake,
    );
    assert_eq!(carol.pending_principal, 0, "carol's pending was promoted on settle",);
    assert_eq!(dave.pending_principal, 0, "dave's pending was promoted on settle",);
}

// =============================================================================
// Sui parity: matched test expectations for proportional rewards
// =============================================================================

/// Sui's `validator_rewards` parity (`rewards_distribution_tests.move`):
/// validators with stake 1:2:3:4 and a 100 SUI reward grow to
/// 125:225:325:425. Per-pool, the equivalent assertion at the pool
/// level: deposit `R` into a single pool, the staker's compound
/// equals `principal × (S + R)/S` for the divisor S at deposit time.
#[test]
fn sui_validator_rewards_per_pool_parity() {
    let mut p = pool();
    p.activation_epoch = Some(0);
    p.active_stake = 100;
    let mut alice = Delegation::new(100, F1_INDEX_SCALE);

    // Match Sui's validator_rewards test: 100 stake, 25 reward
    // (since a 4-way 100 split with 1:2:3:4 ratio gives validator 1
    // the share 25). Expected post-fold compound: 125.
    p.advance_epoch(25);

    auto_settle(&mut alice, &p, 1);
    assert_eq!(alice.principal, 125, "Sui validator_rewards: 100→125 with 25 reward");
}

/// Regression for the additive→multiplicative migration: a single
/// staker who stays settled-stale across multiple epochs must end up
/// with the same principal as if they had settled every boundary.
/// Under the prior additive index this test failed by ~9.5M shannons
/// in the 1B-stake / 200M-reward / 100M-reward sequence.
#[test]
fn settle_stale_matches_settle_every_epoch() {
    // Two pools, identical reward stream, one settled every epoch
    // and one settled only at the end. The end-state principals
    // must agree within rounding.
    let mut p_eager = pool();
    p_eager.activation_epoch = Some(0);
    p_eager.active_stake = 1_000_000_000;

    let mut p_lazy = pool();
    p_lazy.activation_epoch = Some(0);
    p_lazy.active_stake = 1_000_000_000;

    let mut alice_eager = Delegation::new(1_000_000_000, F1_INDEX_SCALE);
    let mut alice_lazy = Delegation::new(1_000_000_000, F1_INDEX_SCALE);

    let rewards = [100_000_000u64, 200_000_000, 50_000_000, 300_000_000];

    for (i, &r) in rewards.iter().enumerate() {
        p_eager.advance_epoch(r);
        p_lazy.advance_epoch(r);
        // Eager settles every epoch.
        auto_settle(&mut alice_eager, &p_eager, (i + 1) as u64);
        // Lazy stays stale until the end.
    }

    auto_settle(&mut alice_lazy, &p_lazy, rewards.len() as u64);

    let diff = alice_eager.principal.abs_diff(alice_lazy.principal);
    assert!(
        diff <= 2,
        "eager vs lazy settle must agree (within rounding), got eager={}, lazy={}, diff={}",
        alice_eager.principal,
        alice_lazy.principal,
        diff,
    );
}
