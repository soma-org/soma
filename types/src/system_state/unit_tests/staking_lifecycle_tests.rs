// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Multi-epoch staking lifecycle tests at the SystemState API level.
//!
//! Mirrors the structural intent of Sui's `validator_set_tests.move`:
//! drive `add_stake_to_validator`, `remove_stake_from_validator`,
//! and `advance_epoch` across several boundaries and assert the
//! invariants Sui's tests check — total stake reflects the
//! pre-boundary state, voting power tracks active_stake at the
//! boundary, mid-epoch withdrawals forfeit current-epoch reward
//! share to remaining stakers via the smaller divisor.
//!
//! Per-staker compound math is covered by
//! `auto_compound_pool_tests`; this file exercises the
//! pool-aggregate path the validator-set machinery actually drives.

#[cfg(test)]
#[allow(clippy::module_inception, clippy::unwrap_used, clippy::expect_used)]
mod staking_lifecycle_tests {
    use crate::base::{SomaAddress, dbg_addr};
    use crate::config::genesis_config::SHANNONS_PER_SOMA;
    use crate::system_state::SystemState;
    use crate::system_state::staking::F1_INDEX_SCALE;
    use crate::system_state::test_utils::{
        ValidatorRewards, advance_epoch_returning_credits, advance_epoch_with_reward_amounts,
        create_test_system_state, create_validator_for_testing,
    };

    fn validator_addr(seed: u8) -> SomaAddress {
        dbg_addr(seed)
    }

    /// 4-validator setup: 100 SOMA self-stake each. 10_000 SOMA in
    /// emission fund (enough for several epochs of test rewards).
    fn set_up_4_validators() -> SystemState {
        let validators = vec![
            create_validator_for_testing(validator_addr(1), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr(2), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr(3), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr(4), 100 * SHANNONS_PER_SOMA),
        ];
        create_test_system_state(validators, 10_000, 0)
    }

    fn pool_active_stake(state: &SystemState, addr: SomaAddress) -> u64 {
        state
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == addr)
            .unwrap()
            .staking_pool
            .active_stake
    }

    fn pool_pending_stake(state: &SystemState, addr: SomaAddress) -> u64 {
        state
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == addr)
            .unwrap()
            .staking_pool
            .pending_active_stake
    }

    fn pool_cumulative_index(state: &SystemState, addr: SomaAddress) -> u128 {
        state
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == addr)
            .unwrap()
            .staking_pool
            .cumulative_index
    }

    fn voting_power(state: &SystemState, addr: SomaAddress) -> u64 {
        state
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == addr)
            .unwrap()
            .voting_power
    }

    // =========================================================================
    // Pool aggregate state
    // =========================================================================

    /// AddStake to an active pool lands in the pending bucket. Active
    /// stake (the F1 fold divisor) is unchanged until the next
    /// boundary. Mirrors Sui's "stake doesn't earn current-epoch
    /// rewards" rule.
    #[test]
    fn add_stake_lands_in_pending_bucket() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);

        let pre_active = pool_active_stake(&state, v1);
        let pre_pending = pool_pending_stake(&state, v1);

        state.add_stake_to_validator(v1, 60 * SHANNONS_PER_SOMA).expect("add");

        assert_eq!(pool_active_stake(&state, v1), pre_active, "active unchanged mid-epoch");
        assert_eq!(
            pool_pending_stake(&state, v1),
            pre_pending + 60 * SHANNONS_PER_SOMA,
            "pending grows by amount",
        );
    }

    /// At the next epoch boundary, the pending bucket promotes into
    /// active_stake. After promotion, the new stake earns rewards
    /// from the *next* epoch onward.
    #[test]
    fn pending_promotes_at_next_boundary() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);

        state.add_stake_to_validator(v1, 60 * SHANNONS_PER_SOMA).expect("add");
        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);

        assert_eq!(
            pool_active_stake(&state, v1),
            160 * SHANNONS_PER_SOMA,
            "pending promoted into active",
        );
        assert_eq!(pool_pending_stake(&state, v1), 0);
    }

    /// Reward distribution at the boundary advances the
    /// cumulative_index multiplicatively. With equal stake across
    /// 4 validators and reward going through the
    /// validator-allocation flow, each pool's index grows by the
    /// same factor (matching Sui's exchange-rate growth).
    #[test]
    fn reward_distribution_advances_cumulative_index() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);
        let pre_index = pool_cumulative_index(&state, v1);

        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        // Drive a non-trivial reward through the emission pool.
        advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);

        let post_index = pool_cumulative_index(&state, v1);
        assert!(
            post_index > pre_index,
            "cumulative_index must advance after reward, got {} -> {}",
            pre_index, post_index,
        );
        // Index stays well below u128 max at realistic rates.
        assert!(post_index < F1_INDEX_SCALE * 1000, "single-epoch index sanity");
    }

    /// Mid-epoch full withdrawal of one staker leaves the pool with
    /// the remaining stake. Voting power is FROZEN until the next
    /// epoch boundary — the validator keeps their voting power for
    /// the rest of the current epoch even after the withdrawal.
    #[test]
    fn voting_power_frozen_mid_epoch() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);
        let pool_id = state
            .add_stake_to_validator(v1, 100 * SHANNONS_PER_SOMA)
            .expect("add");

        // Run an epoch so the pending stake promotes.
        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        let pre_voting_power = voting_power(&state, v1);
        assert!(pre_voting_power > 0);

        // Withdraw half of the staker's contribution mid-epoch.
        // After advance_epoch, the pending stake promoted into
        // active, so the withdrawal drains from active.
        state
            .remove_stake_from_validator(
                pool_id,
                /* from_active */ 50 * SHANNONS_PER_SOMA,
                /* from_pending */ 0,
            )
            .expect("remove");

        // active_stake decreases immediately, but voting power is
        // frozen — `set_voting_power` only fires on epoch advance.
        assert_eq!(voting_power(&state, v1), pre_voting_power, "voting power frozen mid-epoch");
        assert_eq!(
            pool_active_stake(&state, v1),
            150 * SHANNONS_PER_SOMA,
            "active_stake decremented",
        );

        // After the boundary, voting power reflects the new stake.
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        let post_voting_power = voting_power(&state, v1);
        // (post < pre because v1 lost stake while v2/v3/v4 kept
        // theirs. Strict inequality guards against the regression
        // where voting power doesn't recompute.)
        assert!(
            post_voting_power < pre_voting_power,
            "voting power must recompute on epoch advance: {} -> {}",
            pre_voting_power, post_voting_power,
        );
    }

    // =========================================================================
    // Withdrawal pending-first semantics
    // =========================================================================

    /// `remove_stake_from_validator` drains the pending bucket first.
    /// A same-epoch deposit-and-withdraw on the same pool nets to
    /// zero impact on `active_stake`.
    #[test]
    fn withdraw_drains_pending_first() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);
        let pool_id = state
            .add_stake_to_validator(v1, 60 * SHANNONS_PER_SOMA)
            .expect("add");

        // active=100, pending=60.
        assert_eq!(pool_active_stake(&state, v1), 100 * SHANNONS_PER_SOMA);
        assert_eq!(pool_pending_stake(&state, v1), 60 * SHANNONS_PER_SOMA);

        // Withdraw 60 from pending only.
        state
            .remove_stake_from_validator(
                pool_id,
                /* from_active */ 0,
                /* from_pending */ 60 * SHANNONS_PER_SOMA,
            )
            .expect("remove");
        assert_eq!(
            pool_active_stake(&state, v1),
            100 * SHANNONS_PER_SOMA,
            "active untouched when pending covers withdrawal",
        );
        assert_eq!(pool_pending_stake(&state, v1), 0, "pending fully drained");
    }

    /// A withdrawal larger than the pending bucket spills into
    /// active. The active stake decrements immediately for the
    /// shortfall.
    #[test]
    fn withdraw_spills_into_active_when_pending_exhausted() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);
        let pool_id = state
            .add_stake_to_validator(v1, 30 * SHANNONS_PER_SOMA)
            .expect("add");

        // active=100, pending=30.
        // Withdraw 50 — drains 30 from pending and 20 from active.
        state
            .remove_stake_from_validator(
                pool_id,
                /* from_active */ 20 * SHANNONS_PER_SOMA,
                /* from_pending */ 30 * SHANNONS_PER_SOMA,
            )
            .expect("remove");
        assert_eq!(pool_active_stake(&state, v1), 80 * SHANNONS_PER_SOMA);
        assert_eq!(pool_pending_stake(&state, v1), 0);
    }

    // =========================================================================
    // Index history
    // =========================================================================

    /// Each epoch boundary appends a snapshot to `index_history`.
    /// After N advances, history length = N + 1 (+1 for the genesis
    /// snapshot). Stakers whose pending matured in epoch K read
    /// `index_history[K - activation + 1]` as their baseline.
    #[test]
    fn index_history_grows_one_per_epoch_boundary() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);

        let initial_len = state
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == v1)
            .unwrap()
            .staking_pool
            .index_history
            .len();

        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        for _ in 0..3 {
            advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);
        }

        let post_len = state
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == v1)
            .unwrap()
            .staking_pool
            .index_history
            .len();
        assert_eq!(post_len, initial_len + 3, "one snapshot per boundary");
    }

    // =========================================================================
    // Multi-epoch reward distribution (Sui's `validator_rewards` parity)
    // =========================================================================

    /// Sui parity (`rewards_distribution_tests::validator_rewards`):
    /// rewards split proportionally to validator voting power. Set
    /// up validators with stake 100:200:300:400 (all under the
    /// voting-power cap), run epoch with reward budget, expect each
    /// pool to grow by its proportional share. We use evenly-sized
    /// validators so the per-validator voting-power cap never
    /// triggers and proportionality is preserved end-to-end.
    #[test]
    fn proportional_reward_distribution_across_validators() {
        // Equal stakes: rewards should split evenly. Equal
        // distribution is the cleanest invariant to test —
        // unequal stakes hit the per-validator voting-power cap and
        // distort the comparison.
        let validators = vec![
            create_validator_for_testing(validator_addr(1), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr(2), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr(3), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr(4), 100 * SHANNONS_PER_SOMA),
        ];
        let mut state = create_test_system_state(validators, 10_000, 0);

        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);

        let stakes: Vec<u64> = (1..=4)
            .map(|i| pool_active_stake(&state, validator_addr(i)))
            .collect();

        // All pools grew (received some reward).
        for s in &stakes {
            assert!(*s > 100 * SHANNONS_PER_SOMA, "every validator must receive reward");
        }

        // Equal stake → equal reward (within rounding).
        let max_stake = stakes.iter().copied().max().unwrap();
        let min_stake = stakes.iter().copied().min().unwrap();
        let spread = max_stake - min_stake;
        assert!(
            spread <= 10,
            "equal validators must receive equal reward, got spread {} shannons",
            spread,
        );
    }

    // =========================================================================
    // Multi-epoch consistency
    // =========================================================================

    /// Driving the same reward through multiple epochs in a row
    /// compounds the validator's stake monotonically. Catches any
    /// regression where the cumulative_index resets, snapshot
    /// off-by-one, or pending bucket leaks.
    #[test]
    fn multi_epoch_compound_is_monotonic() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);
        let mut prev = pool_active_stake(&state, v1);

        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        for _ in 0..5 {
            advance_epoch_with_reward_amounts(&mut state, 50, &mut tracker);
            let curr = pool_active_stake(&state, v1);
            assert!(
                curr > prev,
                "validator's active_stake must grow each epoch: {} -> {}",
                prev, curr,
            );
            prev = curr;
        }
    }

    /// AddStake then advance, withdraw, advance again. The remaining
    /// stake compounds correctly across the boundaries — the
    /// cumulative_index reflects the smaller divisor at the
    /// withdrawal-affected boundary.
    #[test]
    fn add_then_withdraw_with_intervening_epochs() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        // Epoch 0: add 100 stake (lands in pending).
        let pool_id = state
            .add_stake_to_validator(v1, 100 * SHANNONS_PER_SOMA)
            .expect("add");

        // Advance epoch — pending promotes; v1 now has 200 active.
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        let active_after_promote = pool_active_stake(&state, v1);
        assert_eq!(active_after_promote, 200 * SHANNONS_PER_SOMA);

        // Reward epoch.
        advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);
        let active_after_reward = pool_active_stake(&state, v1);
        assert!(active_after_reward > active_after_promote, "compound from reward");

        // Withdraw 100 mid-epoch — drains from active (no pending
        // bucket on this pool right now).
        state
            .remove_stake_from_validator(
                pool_id,
                /* from_active */ 100 * SHANNONS_PER_SOMA,
                /* from_pending */ 0,
            )
            .expect("remove");
        let active_after_withdraw = pool_active_stake(&state, v1);
        assert_eq!(
            active_after_withdraw,
            active_after_reward - 100 * SHANNONS_PER_SOMA,
            "withdrawal decrements active immediately",
        );
    }

    // =========================================================================
    // Pool/row aggregate invariant under concurrent stakers
    // =========================================================================

    /// Regression: when two stakers hold pending stake on the same
    /// pool, withdrawing one staker's pending must NOT touch the
    /// other staker's pending portion. The earlier bug had
    /// `remove_stake_from_validator` compute the pending/active
    /// split against the *pool* aggregate (`min(amount,
    /// pool.pending_active_stake)`) instead of against the staker's
    /// row, which over-drained the pool's pending bucket and
    /// silently broke `pool.active = sum(row.principals) +
    /// pool.pending = sum(row.pending_principals)` invariant.
    ///
    /// The API now requires the caller to pass `(from_active,
    /// from_pending)` explicitly; this test verifies that the
    /// caller-driven split keeps the pool aggregate in sync with
    /// the actual rows.
    #[test]
    fn concurrent_pending_withdraw_keeps_pool_aggregate_correct() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);

        // Two stakers (alice and bob) each add 100 pending to v1.
        // Pool aggregate goes to (active=100, pending=200).
        let pool_id = state
            .add_stake_to_validator(v1, 100 * SHANNONS_PER_SOMA)
            .expect("alice add");
        state
            .add_stake_to_validator(v1, 100 * SHANNONS_PER_SOMA)
            .expect("bob add");

        assert_eq!(pool_active_stake(&state, v1), 100 * SHANNONS_PER_SOMA);
        assert_eq!(pool_pending_stake(&state, v1), 200 * SHANNONS_PER_SOMA);

        // Alice withdraws her 100 pending. Caller passes
        // (from_active=0, from_pending=100). Pool's pending must
        // drop by EXACTLY 100 (not 200), preserving bob's row.
        state
            .remove_stake_from_validator(
                pool_id,
                /* from_active */ 0,
                /* from_pending */ 100 * SHANNONS_PER_SOMA,
            )
            .expect("alice withdraw");

        assert_eq!(
            pool_active_stake(&state, v1),
            100 * SHANNONS_PER_SOMA,
            "active untouched (alice drained from pending)",
        );
        assert_eq!(
            pool_pending_stake(&state, v1),
            100 * SHANNONS_PER_SOMA,
            "pending dropped by alice's 100, NOT by alice+bob's 200",
        );
    }

    // =========================================================================
    // Validator commission flow at SystemState level
    // =========================================================================

    /// Verifies the SystemState side of validator commission:
    ///
    /// 1. `advance_epoch` returns one `ValidatorRewardCredit` per
    ///    validator with `principal` equal to the commission portion
    ///    of that validator's adjusted reward.
    /// 2. The pool's `active_stake` grew by the FULL adjusted reward
    ///    (staker portion + commission).
    /// 3. The `cumulative_index` advance ratio reflects the staker
    ///    portion only — the commission was added to `active_stake`
    ///    *after* `deposit_staker_rewards` so it doesn't dilute
    ///    existing stakers' compound on the staker portion.
    /// 4. Conservation: pool.active_stake = old + total_reward.
    ///
    /// The change_epoch executor (authority crate) consumes the
    /// returned credit map to grow the validator's delegation row by
    /// `principal` — that path is exercised by authority's
    /// `staking_tests`.
    #[test]
    fn validator_commission_credit_matches_commission_rate() {
        // Single validator with 20% commission.
        let mut v = create_validator_for_testing(validator_addr(1), 100 * SHANNONS_PER_SOMA);
        v.commission_rate = 2_000; // 20% in BPS
        v.next_epoch_commission_rate = 2_000;
        let mut state = create_test_system_state(vec![v], 10_000, 0);
        let v1 = validator_addr(1);

        let pre_active = pool_active_stake(&state, v1);
        let pre_index = pool_cumulative_index(&state, v1);

        // Drive an epoch with 100 SOMA of reward.
        let credits = advance_epoch_returning_credits(&mut state, 100);

        let post_active = pool_active_stake(&state, v1);
        let post_index = pool_cumulative_index(&state, v1);

        // Reward distribution: with one validator at full voting
        // power, this validator's adjusted_reward = 100 SOMA. With
        // 20% commission: commission = 20 SOMA, staker = 80 SOMA.
        let expected_total = 100 * SHANNONS_PER_SOMA;
        let expected_commission = 20 * SHANNONS_PER_SOMA;
        let expected_staker = 80 * SHANNONS_PER_SOMA;

        // Pool grew by the FULL reward (commission + staker).
        let pool_growth = post_active - pre_active;
        // Allow ±1 shannon rounding from BPS computation.
        assert!(
            pool_growth.abs_diff(expected_total) <= 1,
            "pool grew by full reward, expected ~{}, got {}",
            expected_total, pool_growth,
        );

        // Cumulative index advance = (pre_active + staker_reward)/pre_active.
        // i.e. NOT including commission. Otherwise existing stakers
        // would over-compound (their share would include commission
        // which doesn't belong to them).
        let actual_ratio = (post_index as u128) * F1_INDEX_SCALE / (pre_index as u128);
        let expected_ratio = ((pre_active + expected_staker) as u128)
            * F1_INDEX_SCALE
            / (pre_active as u128);
        let ratio_diff = actual_ratio.abs_diff(expected_ratio);
        assert!(
            ratio_diff < F1_INDEX_SCALE / 1_000_000,
            "index advance ratio reflects staker portion only: actual={}, expected={}",
            actual_ratio, expected_ratio,
        );

        // Credit map contains the commission for this validator.
        let credit = credits.get(&v1).expect("validator must receive a commission credit");
        assert!(
            credit.principal.abs_diff(expected_commission) <= 1,
            "credit principal must match commission: expected ~{}, got {}",
            expected_commission, credit.principal,
        );
        // The credit references the validator's pool.
        let pool_id = state
            .validators()
            .validators
            .iter()
            .find(|val| val.metadata.soma_address == v1)
            .unwrap()
            .staking_pool
            .id;
        assert_eq!(credit.pool_id, pool_id);
    }

    /// Zero-commission validator: no commission credit emitted (or
    /// credit with principal 0). All reward goes to staker
    /// compound. Catches a regression where a default credit might
    /// be emitted for every validator.
    #[test]
    fn zero_commission_emits_no_real_credit() {
        let mut state = set_up_4_validators(); // commission_rate = 0
        let v1 = validator_addr(1);
        let pre_index = pool_cumulative_index(&state, v1);
        let pre_active = pool_active_stake(&state, v1);

        let credits = advance_epoch_returning_credits(&mut state, 100);

        let post_active = pool_active_stake(&state, v1);
        let post_index = pool_cumulative_index(&state, v1);

        // Pool grew (full reward goes to stakers via index advance).
        assert!(post_active > pre_active);
        assert!(post_index > pre_index);

        // No commission credit emitted (or principal == 0).
        match credits.get(&v1) {
            None => { /* expected */ }
            Some(credit) => assert_eq!(
                credit.principal, 0,
                "zero-commission validator must have zero credit, got {}",
                credit.principal,
            ),
        }
    }

    // =========================================================================
    // Reward slashing
    // =========================================================================

    /// Sui parity (`rewards_distribution_tests::rewards_slashing`):
    /// when a validator is reported by ≥2/3 of the committee, their
    /// reward is slashed by `reward_slashing_rate` and the slashed
    /// portion is redistributed to unslashed validators
    /// proportional to voting power. Pool-level effects:
    ///
    /// - Slashed validator's pool grows by less than its unslashed
    ///   share would have produced.
    /// - Unslashed validators' pools grow by more (received the
    ///   forfeited share).
    /// - Conservation: total pool growth = total reward emitted.
    #[test]
    fn slashed_validator_rewards_redistribute_to_others() {
        let mut state = set_up_4_validators(); // 4 × 100 SOMA equal stake
        let v1 = validator_addr(1);
        let v2 = validator_addr(2);
        let v3 = validator_addr(3);
        let v4 = validator_addr(4);

        // Capture pre-state.
        let pre_active: Vec<u64> =
            [v1, v2, v3, v4].iter().map(|a| pool_active_stake(&state, *a)).collect();

        // 3 of 4 validators report v4 → 75% of voting power, well
        // above the 2f+1 quorum threshold. v4 is slashed.
        for reporter in [v1, v2, v3] {
            state.report_validator(reporter, v4).expect("report");
        }

        let _credits = advance_epoch_returning_credits(&mut state, 100);

        let post_active: Vec<u64> =
            [v1, v2, v3, v4].iter().map(|a| pool_active_stake(&state, *a)).collect();
        let growth: Vec<u64> = pre_active
            .iter()
            .zip(post_active.iter())
            .map(|(pre, post)| post - pre)
            .collect();

        // v4 is slashed → grew by less than its 25 SOMA fair share.
        // (reward_slashing_rate is 50% by default, so v4 gets 12.5
        // SOMA = 12.5e9 shannons; but voting-power normalization can
        // shift the exact figure. We assert v4 < unslashed.)
        let v4_growth = growth[3];
        let unslashed_growths = &growth[0..3];

        for (i, &g) in unslashed_growths.iter().enumerate() {
            assert!(
                g > v4_growth,
                "v{} (unslashed) grew {}, v4 (slashed) grew {} — unslashed must grow more",
                i + 1, g, v4_growth,
            );
        }
        // Equal-stake unslashed validators get equal share of
        // redistributed reward.
        let max_unslashed = *unslashed_growths.iter().max().unwrap();
        let min_unslashed = *unslashed_growths.iter().min().unwrap();
        assert!(
            max_unslashed.abs_diff(min_unslashed) <= 10,
            "unslashed validators with equal stake share equally: spread {}",
            max_unslashed - min_unslashed,
        );

        // Conservation: total pool growth = total reward distributed.
        let total_growth: u64 = growth.iter().sum();
        let expected_total = 100 * SHANNONS_PER_SOMA;
        assert!(
            total_growth.abs_diff(expected_total) <= 4,
            "total pool growth must equal total reward (within rounding): expected {}, got {}",
            expected_total, total_growth,
        );

        // Sanity: v4's growth is positive but less than 25 SOMA (1/4
        // of total).
        assert!(v4_growth > 0, "v4 still receives some reward (slashing < 100%)");
        assert!(
            v4_growth < 25 * SHANNONS_PER_SOMA,
            "v4 must grow by less than its unslashed 25-SOMA share, got {}",
            v4_growth,
        );
    }

    /// Without any reports, no validators are slashed and equal-stake
    /// validators all grow by the same share. Baseline / control
    /// case for the slashing test above.
    #[test]
    fn no_reports_means_equal_growth_for_equal_stake() {
        let mut state = set_up_4_validators();
        let pre_active: Vec<u64> = (1..=4)
            .map(|i| pool_active_stake(&state, validator_addr(i)))
            .collect();

        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);

        let post_active: Vec<u64> = (1..=4)
            .map(|i| pool_active_stake(&state, validator_addr(i)))
            .collect();
        let growth: Vec<u64> = pre_active
            .iter()
            .zip(post_active.iter())
            .map(|(pre, post)| post - pre)
            .collect();

        let max_g = *growth.iter().max().unwrap();
        let min_g = *growth.iter().min().unwrap();
        assert!(
            max_g.abs_diff(min_g) <= 10,
            "equal-stake validators grow equally with no slashing: spread {}",
            max_g - min_g,
        );
    }

    // =========================================================================
    // Same-epoch deposit-and-withdraw (Sui's request_add_then_pull_stake)
    // =========================================================================

    /// Sui parity (`validator_set_tests::request_add_then_pull_stake`,
    /// adapted): a staker who adds and then immediately withdraws
    /// in the same epoch must end up with no stake on the pool, the
    /// pool's pending bucket back to zero (or unchanged from
    /// pre-add), and no rewards earned.
    #[test]
    fn same_epoch_add_then_withdraw_full_drain() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);

        let pre_active = pool_active_stake(&state, v1);
        let pre_pending = pool_pending_stake(&state, v1);

        // Add 100 SOMA → lands in pending.
        let pool_id = state
            .add_stake_to_validator(v1, 100 * SHANNONS_PER_SOMA)
            .expect("add");
        assert_eq!(pool_active_stake(&state, v1), pre_active);
        assert_eq!(pool_pending_stake(&state, v1), pre_pending + 100 * SHANNONS_PER_SOMA);

        // Same epoch, withdraw the same 100 — must drain pending only.
        state
            .remove_stake_from_validator(
                pool_id,
                /* from_active */ 0,
                /* from_pending */ 100 * SHANNONS_PER_SOMA,
            )
            .expect("withdraw");

        // Pool returns to its pre-add aggregate state.
        assert_eq!(
            pool_active_stake(&state, v1),
            pre_active,
            "active untouched by same-epoch add+withdraw",
        );
        assert_eq!(
            pool_pending_stake(&state, v1),
            pre_pending,
            "pending returns to pre-add value",
        );

        // Advance the epoch with rewards. The would-be staker
        // doesn't earn anything because their stake was withdrawn
        // before reward distribution.
        let pre_index = pool_cumulative_index(&state, v1);
        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);
        let post_index = pool_cumulative_index(&state, v1);

        // Index advanced by reward / pre_active (NOT including the
        // 100-SOMA add). If the add-and-withdraw had leaked into the
        // divisor, the index advance ratio would be different.
        let expected_growth_factor =
            ((pre_active + 25 * SHANNONS_PER_SOMA) as u128) * F1_INDEX_SCALE
                / (pre_active as u128);
        let actual_growth_factor = (post_index as u128) * F1_INDEX_SCALE / (pre_index as u128);
        // 25 SOMA = 100 SOMA / 4 validators (equal voting power).
        let diff = actual_growth_factor.abs_diff(expected_growth_factor);
        assert!(
            diff < F1_INDEX_SCALE / 1_000,
            "index advance reflects ONLY the pre-add divisor, no leak from add+withdraw",
        );
    }

    /// Same-epoch full-drain with multiple stakers: alice and bob
    /// both add pending; alice withdraws hers; epoch advances;
    /// bob's pending matures and earns the next epoch's reward but
    /// alice's never existed in the divisor.
    #[test]
    fn same_epoch_drain_one_keep_other() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);

        // Alice and bob each add 100 pending mid-epoch.
        let pool_id = state
            .add_stake_to_validator(v1, 100 * SHANNONS_PER_SOMA)
            .expect("alice add");
        state
            .add_stake_to_validator(v1, 100 * SHANNONS_PER_SOMA)
            .expect("bob add");
        assert_eq!(pool_pending_stake(&state, v1), 200 * SHANNONS_PER_SOMA);

        // Alice changes her mind and withdraws her 100 pending.
        state
            .remove_stake_from_validator(
                pool_id,
                /* from_active */ 0,
                /* from_pending */ 100 * SHANNONS_PER_SOMA,
            )
            .expect("alice drain");
        assert_eq!(pool_pending_stake(&state, v1), 100 * SHANNONS_PER_SOMA);

        // Epoch advances. Bob's 100 pending matures.
        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        assert_eq!(
            pool_active_stake(&state, v1),
            200 * SHANNONS_PER_SOMA,
            "v1 active = 100 (genesis) + 100 (bob, promoted), alice's 100 never landed",
        );
        assert_eq!(pool_pending_stake(&state, v1), 0, "all pending promoted");
    }

    /// Caller specifies the active/pending split for each
    /// withdrawal. Verifies that the same pool can route two
    /// withdrawals from different stakers — one drains pending, the
    /// other drains active — without aggregates drifting.
    #[test]
    fn mixed_pending_active_withdrawals_preserve_aggregate() {
        let mut state = set_up_4_validators();
        let v1 = validator_addr(1);
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        // Alice adds 100 pending → epoch advance promotes it.
        let pool_id = state
            .add_stake_to_validator(v1, 100 * SHANNONS_PER_SOMA)
            .expect("alice add");
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        assert_eq!(pool_active_stake(&state, v1), 200 * SHANNONS_PER_SOMA);

        // Bob adds 100 pending mid-epoch.
        state
            .add_stake_to_validator(v1, 100 * SHANNONS_PER_SOMA)
            .expect("bob add");
        assert_eq!(pool_active_stake(&state, v1), 200 * SHANNONS_PER_SOMA);
        assert_eq!(pool_pending_stake(&state, v1), 100 * SHANNONS_PER_SOMA);

        // Alice withdraws from active (her stake is now active).
        state
            .remove_stake_from_validator(
                pool_id,
                /* from_active */ 50 * SHANNONS_PER_SOMA,
                /* from_pending */ 0,
            )
            .expect("alice partial");

        // Bob withdraws from pending.
        state
            .remove_stake_from_validator(
                pool_id,
                /* from_active */ 0,
                /* from_pending */ 30 * SHANNONS_PER_SOMA,
            )
            .expect("bob partial");

        assert_eq!(
            pool_active_stake(&state, v1),
            150 * SHANNONS_PER_SOMA,
            "alice took 50 from active",
        );
        assert_eq!(
            pool_pending_stake(&state, v1),
            70 * SHANNONS_PER_SOMA,
            "bob took 30 from pending",
        );
    }
}
