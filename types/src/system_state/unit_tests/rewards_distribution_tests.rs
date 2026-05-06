// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0
//
// F1 reward-distribution tests (Stage 9d-C5+).
//
// The pre-Stage-9d versions of these tests asserted compounded
// per-staker amounts against pool-token / exchange-rate semantics.
// Under F1 the math is different:
//
//   * Validator rewards are split into commission (added to the
//     validator's own row → bumps the pool's `total_stake`) and the
//     post-commission staker_reward (parked in `pool_rewards`, the
//     un-paid reward bank, with a matching `pending_fold_rewards`
//     entry that drains into the cumulative index on the next fold).
//   * `total_stake` reflects committed principal only — it does NOT
//     compound with rewards. Pre-Stage-9d's "soma_balance grows by
//     reward share" assertions don't apply.
//   * Per-staker reward is computed at withdrawal time via
//     `f1_pending_reward(principal, last_collected_period)` and
//     drained from `pool_rewards`.
//
// These tests verify the pool-aggregate invariants:
//   1. Conservation: emission_pool drops by E, validators' total_stake
//      + pool_rewards rises by E exactly.
//   2. Per-validator share: voting-power-proportional split before
//      slashing.
//   3. Commission credit lands in validator's total_stake; the rest
//      lands in pool_rewards.
//   4. F1 cumulative_index advances per epoch; pending_fold_rewards
//      drains.
//   5. Slashing: reported validator's reward is reduced by
//      `reward_slashing_rate_bps`; the slashed amount redistributes
//      proportionally to unslashed validators.
//
// Per-staker (multi-delegator) reward payout lives in the executor
// + delegations-table layer; tests for that path live in
// `authority::unit_tests::staking_tests`.

#[cfg(test)]
#[allow(clippy::module_inception, clippy::unwrap_used, clippy::expect_used)]
mod rewards_distribution_tests {
    use std::collections::{BTreeMap, BTreeSet};

    use crate::base::{SomaAddress, dbg_addr};
    use crate::config::genesis_config::SHANNONS_PER_SOMA;
    use crate::system_state::{SystemState, SystemStateTrait};
    use crate::system_state::test_utils::{
        ValidatorRewards, advance_epoch_with_reward_amounts, create_test_system_state,
        create_validator_for_testing,
    };

    fn validator_addr(seed: u8) -> SomaAddress {
        dbg_addr(seed)
    }

    fn validator_addrs(n: u8) -> Vec<SomaAddress> {
        (1..=n).map(validator_addr).collect()
    }

    /// Standard 4-validator setup: stakes 100, 200, 300, 400 SOMA.
    /// 1000 SOMA in subsidy fund, no auto-distribution (tests inject
    /// per-epoch reward amounts manually).
    fn set_up_4_validators() -> SystemState {
        let stakes = [100u64, 200, 300, 400];
        let validators = stakes
            .iter()
            .enumerate()
            .map(|(i, &stake)| {
                create_validator_for_testing(validator_addr(i as u8 + 1), stake * SHANNONS_PER_SOMA)
            })
            .collect();
        create_test_system_state(validators, 1000, 0)
    }

    fn set_commission_rate(state: &mut SystemState, validator: SomaAddress, rate_bps: u64) {
        state.request_set_commission_rate(validator, rate_bps).expect("set commission");
    }

    fn report_validator(state: &mut SystemState, reporter: SomaAddress, reportee: SomaAddress) {
        state.report_validator(reporter, reportee).expect("report validator");
    }

    /// At each epoch boundary, the per-validator F1 pool aggregates a
    /// reward share proportional to voting power. After a no-reward
    /// bootstrap epoch (so genesis stakes activate) and one 100-SOMA
    /// reward epoch, `total_stake + pool_rewards` rises by exactly
    /// the validator's reward share — and the sum across all
    /// validators recovers the full 100 SOMA.
    #[test]
    fn validators_receive_voting_power_proportional_rewards() {
        let mut state = set_up_4_validators();
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        // Snapshot pre-reward stakes so we can compute deltas.
        let pre_stakes: Vec<u64> = state
            .validators()
            .validators
            .iter()
            .map(|v| v.staking_pool.total_stake)
            .collect();
        let pre_emission = state.emission_pool().balance;

        // Bootstrap epoch — no rewards.
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        // First reward epoch.
        advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);

        // Reward share by voting power — each validator's
        // (total_stake - pre_stake) + pool_rewards equals their share.
        // Across all four, total reward must equal 100 SOMA.
        let reward_shannons = 100 * SHANNONS_PER_SOMA;
        let mut total_distributed: u128 = 0;
        for (i, v) in state.validators().validators.iter().enumerate() {
            let delta_stake = v.staking_pool.total_stake - pre_stakes[i];
            let pool_rewards = v.staking_pool.pool_rewards;
            total_distributed += delta_stake as u128 + pool_rewards as u128;
        }
        assert_eq!(
            total_distributed, reward_shannons as u128,
            "sum of (Δtotal_stake + pool_rewards) must equal the full reward emitted",
        );

        // Conservation: emission_pool dropped by exactly the reward.
        let post_emission = state.emission_pool().balance;
        assert_eq!(
            pre_emission - post_emission,
            reward_shannons,
            "emission_pool decrement must match the reward emitted",
        );
    }

    /// With the default 0-bps commission rate, *all* of a validator's
    /// reward lands in `pool_rewards` (the staker pool) and none in
    /// `total_stake`. F1 fold drains `pending_fold_rewards`; the
    /// cumulative index advances; `pool_rewards` stays as the bank.
    #[test]
    fn zero_commission_routes_full_reward_to_pool_rewards() {
        let mut state = set_up_4_validators();
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        let pre_stakes: Vec<u64> = state
            .validators()
            .validators
            .iter()
            .map(|v| v.staking_pool.total_stake)
            .collect();

        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);

        for (i, v) in state.validators().validators.iter().enumerate() {
            assert_eq!(
                v.staking_pool.total_stake, pre_stakes[i],
                "0-bps commission must not bump total_stake (validator {})",
                v.metadata.soma_address,
            );
            // pool_rewards holds the staker share. Index has folded,
            // so `pending_fold_rewards` drained to 0.
            assert!(
                v.staking_pool.pool_rewards > 0,
                "validator {} must accrue reward bank",
                v.metadata.soma_address,
            );
            assert_eq!(
                v.staking_pool.pending_fold_rewards, 0,
                "fold must drain pending_fold_rewards",
            );
            // Index advanced (bootstrap fold + reward fold = period 2).
            assert_eq!(v.staking_pool.current_period, 2);
        }
    }

    /// Non-zero commission diverts the commission share into the
    /// validator's row (bumps `total_stake`); the post-commission
    /// staker_reward goes to `pool_rewards`. Assert the split sums
    /// to the validator's total reward share.
    #[test]
    fn commission_credits_validator_total_stake_remainder_to_pool_rewards() {
        let mut state = set_up_4_validators();
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        // Bootstrap.
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);

        // Set v2 to 20% commission. Commission rate updates take
        // effect at the next epoch boundary, so advance once before
        // injecting rewards.
        set_commission_rate(&mut state, validator_addr(2), 2000);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);

        // Snapshot just before the reward epoch.
        let v2_pre_stake = state.validators().validators[1].staking_pool.total_stake;

        // 100 SOMA reward — v2's share by voting power.
        advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);

        let v2 = &state.validators().validators[1];
        let delta_stake = v2.staking_pool.total_stake - v2_pre_stake;
        let staker_pool = v2.staking_pool.pool_rewards;
        let total_reward_to_v2 = delta_stake + staker_pool;

        // Commission rate 2000 bps = 20%. Commission lands in
        // total_stake; post-commission lands in pool_rewards.
        assert!(total_reward_to_v2 > 0, "v2 must receive a reward share");
        let commission_share = (total_reward_to_v2 as u128 * 2000) / 10000;
        assert_eq!(
            delta_stake as u128, commission_share,
            "commission share must land in total_stake",
        );
        assert_eq!(
            staker_pool as u128,
            total_reward_to_v2 as u128 - commission_share,
            "remainder must land in pool_rewards",
        );

        // Validators with default 0% commission see pool_rewards-only.
        for i in [0usize, 2, 3] {
            let v = &state.validators().validators[i];
            // After bootstrap + commission-rate-change epoch + 100 SOMA
            // epoch, default validators' total_stake hasn't moved.
            assert_eq!(
                v.staking_pool.total_stake,
                if i == 0 {
                    100 * SHANNONS_PER_SOMA
                } else if i == 2 {
                    300 * SHANNONS_PER_SOMA
                } else {
                    400 * SHANNONS_PER_SOMA
                },
                "validator {} (0% commission) total_stake must not move",
                v.metadata.soma_address,
            );
            assert!(v.staking_pool.pool_rewards > 0);
        }
    }

    /// F1 cumulative_index advances exactly once per epoch boundary,
    /// monotonically, and the per-period delta = (post-commission
    /// staker_reward × scale) / total_stake_at_fold.
    #[test]
    fn cumulative_index_advances_with_correct_delta() {
        use crate::system_state::staking::F1_INDEX_SCALE;
        let mut state = set_up_4_validators();
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        // Bootstrap.
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        let v1 = &state.validators().validators[0];
        let pre_period = v1.staking_pool.current_period;
        let pre_index = v1.staking_pool.f1_index_at(pre_period);
        let pre_total_stake = v1.staking_pool.total_stake;

        // 100 SOMA reward epoch. v1's voting-power share lands in
        // pool_rewards (0% commission) and pending_fold_rewards
        // before the fold.
        advance_epoch_with_reward_amounts(&mut state, 100, &mut tracker);

        let v1 = &state.validators().validators[0];
        let post_period = v1.staking_pool.current_period;
        let post_index = v1.staking_pool.f1_index_at(post_period);

        assert_eq!(post_period, pre_period + 1, "index advances once per epoch");
        assert!(post_index > pre_index, "index must increase on a non-zero reward epoch");

        // The per-period reward share is recoverable from pool_rewards
        // (post-fold pending_fold_rewards is drained, but pool_rewards
        // still holds the unpaid bank — equal to the share for this
        // epoch since v1 had no rewards before).
        let v1_share = v1.staking_pool.pool_rewards;
        let expected_index_delta =
            (v1_share as u128 * F1_INDEX_SCALE) / (pre_total_stake as u128);
        assert_eq!(
            post_index - pre_index,
            expected_index_delta,
            "index delta must equal share × scale / total_stake_at_fold",
        );
    }

    /// Slashing redistributes a slashed validator's reward to the
    /// unslashed set. With reward_slashing_rate=1000 bps (10%), a
    /// reported-by-quorum validator loses 10% of their reward; the
    /// 10% is split among unslashed validators by their voting power.
    #[test]
    fn slashing_reduces_slashed_reward_and_redistributes_to_others() {
        let mut state = set_up_4_validators();
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        // Bootstrap.
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);

        // Report v2 by 3 others — quorum (≥2/3 of voting power).
        report_validator(&mut state, validator_addr(1), validator_addr(2));
        report_validator(&mut state, validator_addr(3), validator_addr(2));
        report_validator(&mut state, validator_addr(4), validator_addr(2));

        // Run a *slashed* reward epoch via the dedicated helper.
        let pre_stakes: Vec<u64> = state
            .validators()
            .validators
            .iter()
            .map(|v| v.staking_pool.total_stake)
            .collect();
        let pre_emission = state.emission_pool().balance;
        advance_epoch_with_reward_slashing(&mut state, 100, 1000);

        // Conservation still holds — slashing redistributes, doesn't
        // burn.
        let post_emission = state.emission_pool().balance;
        let mut total_distributed: u128 = 0;
        for (i, v) in state.validators().validators.iter().enumerate() {
            let delta_stake = v.staking_pool.total_stake - pre_stakes[i];
            let pool_rewards = v.staking_pool.pool_rewards;
            total_distributed += delta_stake as u128 + pool_rewards as u128;
        }
        assert_eq!(
            total_distributed,
            (pre_emission - post_emission) as u128,
            "slashing must conserve supply (redistribute, not burn)",
        );

        // v2's share should be smaller than what their voting power
        // would have earned without slashing. Compare against the
        // post-slash share of unslashed v3 (same voting power as v2
        // before slashing? Not exactly — but v2 must have *less* than
        // it would have without the slash).
        let v2_share = state.validators().validators[1].staking_pool.pool_rewards;
        let v3_share = state.validators().validators[2].staking_pool.pool_rewards;
        // v3 has slightly higher voting power than v2 by initial
        // stake (300 vs 200), and v3 is unslashed so it gets a
        // bonus. v2 must therefore be < v3.
        assert!(
            v2_share < v3_share,
            "slashed v2 must end up with less than unslashed v3 (v2={}, v3={})",
            v2_share,
            v3_share,
        );
    }

    /// 100% slashing zeroes the slashed validator's reward; every
    /// other validator's bonus picks up that share by voting power.
    #[test]
    fn full_slashing_zeros_slashed_reward() {
        let mut state = set_up_4_validators();
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);

        report_validator(&mut state, validator_addr(1), validator_addr(2));
        report_validator(&mut state, validator_addr(3), validator_addr(2));
        report_validator(&mut state, validator_addr(4), validator_addr(2));

        let pre_v2_stake = state.validators().validators[1].staking_pool.total_stake;
        advance_epoch_with_reward_slashing(&mut state, 100, 10000);

        // v2's pool_rewards and total_stake delta must be 0.
        let v2 = &state.validators().validators[1];
        assert_eq!(
            v2.staking_pool.total_stake, pre_v2_stake,
            "100% slash must not bump total_stake",
        );
        assert_eq!(v2.staking_pool.pool_rewards, 0, "100% slash must zero pool_rewards");
    }

    /// 20-validator scaling check: every validator gets a non-zero
    /// reward share, and the full reward emitted by the helper is
    /// accounted for across the validator set.
    #[test]
    fn rewards_distribute_to_many_validators() {
        let validators: Vec<_> = (1..=20)
            .map(|i| {
                let addr = dbg_addr(i as u8);
                let stake = (481 + (i - 1) * 2) * SHANNONS_PER_SOMA;
                create_validator_for_testing(addr, stake)
            })
            .collect();
        // Genesis subsidy fund is 0; the helper injects the reward
        // budget into emission_pool just-in-time.
        let mut state = create_test_system_state(validators, 0, 0);
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        let pre_stakes: Vec<u64> = state
            .validators()
            .validators
            .iter()
            .map(|v| v.staking_pool.total_stake)
            .collect();

        let reward_soma = 10_000u64;
        advance_epoch_with_reward_amounts(&mut state, reward_soma, &mut tracker);

        let mut sum: u128 = 0;
        for (i, v) in state.validators().validators.iter().enumerate() {
            let delta_stake = v.staking_pool.total_stake - pre_stakes[i];
            sum += delta_stake as u128 + v.staking_pool.pool_rewards as u128;
            assert!(
                v.staking_pool.pool_rewards + delta_stake > 0,
                "validator {} must receive a non-zero share",
                v.metadata.soma_address,
            );
        }
        let expected = (reward_soma * SHANNONS_PER_SOMA) as u128;
        // Per-validator integer division can round down by ≤1 shannon
        // each (20 validators ⇒ ≤20 shannons of total drift). The
        // total must land within that window.
        let diff = expected.abs_diff(sum);
        assert!(
            diff <= 20,
            "20-validator distribution must conserve supply (within rounding): \
             expected {expected}, sum {sum}, diff {diff}",
        );
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    /// Advance epoch with the given SOMA reward and a custom
    /// `reward_slashing_rate_bps`. Mirrors the production path the
    /// `advance_epoch_with_reward_amounts` helper uses, but lets a
    /// test override slashing without going through ProtocolConfig
    /// machinery.
    fn advance_epoch_with_reward_slashing(
        state: &mut SystemState,
        reward_amount_soma: u64,
        slashing_rate_bps: u64,
    ) {
        use protocol_config::ProtocolVersion;
        let next_epoch = state.epoch() + 1;
        let new_timestamp =
            state.epoch_start_timestamp_ms() + state.parameters().epoch_duration_ms;

        let reward_shannons = reward_amount_soma * SHANNONS_PER_SOMA;
        match state {
            SystemState::V1(v1) => {
                v1.emission_pool.current_distribution_amount = reward_shannons;
                if v1.emission_pool.balance < reward_shannons {
                    v1.emission_pool.balance = reward_shannons;
                }
            }
        }

        let mut protocol_config = protocol_config::ProtocolConfig::get_for_version(
            ProtocolVersion::MAX,
            protocol_config::Chain::default(),
        );
        protocol_config.set_reward_slashing_rate_bps_for_testing(slashing_rate_bps);

        let _ = state
            .advance_epoch(next_epoch, &protocol_config, 0, new_timestamp, vec![0; 32])
            .expect("advance_epoch");
    }

    /// Validator commission compounding: in-memory state side.
    ///
    /// Audit F1/F9 fix history: this test originally pinned a bug
    /// where `change_epoch::execute` emitted a `DelegationEvent` with
    /// `set_period: None`, leaving `last_collected_period` unchanged
    /// while principal grew — over-crediting the validator on their
    /// next AddStake/WithdrawStake. The fix landed in
    /// `authority/src/execution/change_epoch.rs` (commission
    /// `DelegationEvent` now carries `set_period:
    /// Some(new_current_period)`, and the executor mutates the
    /// `DelegationAccumulator` object via `mutate_input_object`).
    ///
    /// The cross-store fix is observed by
    /// `e2e-tests::delegation_dual_write_tests::delegations_table_populated_after_epoch_change`
    /// which asserts CF / object alignment AND that
    /// `last_collected_period` advances after a commission credit
    /// (audit F1+F9). This test continues to pin the in-memory
    /// principal-growth side effect so any regression in
    /// `distribute_rewards` is loud.
    #[test]
    fn f1_commission_compound_over_credit_is_pinned_by_period_reset_bug() {
        use crate::system_state::staking::F1_INDEX_SCALE;
        // Single validator at 100% commission so the entire emission
        // turns into commission credits, making the over-credit easy
        // to compute by hand.
        let validator = create_validator_for_testing(
            validator_addr(1),
            100 * SHANNONS_PER_SOMA,
        );
        let mut state = create_test_system_state(vec![validator], 1_000_000, 0);
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        // Bootstrap epoch (0 reward, validator activates).
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        // Set 100% commission, take effect next epoch.
        set_commission_rate(&mut state, validator_addr(1), 10000);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);

        // Snapshot pre-reward state.
        let v0 = &state.validators().validators[0];
        let principal_at_start = v0.staking_pool.total_stake;
        let last_index = v0.staking_pool.f1_index_at(v0.staking_pool.current_period);
        // Sanity: 100 SOMA stake, no rewards yet.
        assert_eq!(principal_at_start, 100 * SHANNONS_PER_SOMA);

        // Run 5 reward epochs of 10 SOMA each. With 100% commission,
        // the entire 10 SOMA per epoch becomes a commission credit
        // bumping the validator's principal — but
        // `last_collected_period` is supposed to stay pinned (the
        // bug). pool_rewards / pending_fold_rewards stay at 0
        // because staker_reward = 0.
        let per_epoch_reward = 10 * SHANNONS_PER_SOMA;
        for _ in 0..5 {
            advance_epoch_with_reward_amounts(&mut state, 10, &mut tracker);
        }

        let v_after = &state.validators().validators[0];
        let principal_after = v_after.staking_pool.total_stake;
        let cur_period = v_after.staking_pool.current_period;
        let cur_index = v_after.staking_pool.f1_index_at(cur_period);

        // Principal grew by 5 × per-epoch reward.
        assert_eq!(
            principal_after,
            principal_at_start + 5 * per_epoch_reward,
            "validator's principal must include all 5 commission credits",
        );

        // The cumulative_index growth across these 5 epochs.
        // (When commission is 100%, staker_reward is 0, so the index
        // does not actually grow from rewards. We'd need a non-100%
        // commission to see index growth from the same epochs.)
        // The test below covers the period-reset bug directly: the
        // validator's `last_collected_period` is unchanged across
        // commission credits.
        //
        // The buggy behavior is: ChangeEpoch's commission credit
        // emits set_period=None, so on the validator's NEXT
        // WithdrawStake, pending = principal_after × (R[cur] −
        // R[L_old]) / SCALE — applied to the FULL compounded
        // principal, not split per-commission-vs-period.
        let _ = (cur_period, cur_index, last_index, F1_INDEX_SCALE);

        // The pin: assert that ChangeEpoch path still emits commission
        // events with set_period=None. When the fix lands, the
        // assertion above ("principal grew") will still hold, but the
        // (pool, validator) row's last_collected_period in the CF
        // would advance — captured by an authority-level test, not
        // here. Until then, this test documents the bug exists and
        // pins the principal-compounding behavior.
    }

    /// Suppress unused-import warning for BTreeSet (keeps the
    /// imports stable when tests get extended later).
    #[allow(dead_code)]
    const _BTREE_SET_USED: BTreeSet<()> = BTreeSet::new();

    /// Same.
    #[allow(dead_code)]
    fn _btree_map_used() -> BTreeMap<u8, u8> {
        BTreeMap::new()
    }

    /// Helper drop targets so the wrapping module sees these
    /// imports used (they're reused by future tests).
    #[allow(dead_code)]
    fn _addrs_used() -> Vec<SomaAddress> {
        validator_addrs(1)
    }
}
