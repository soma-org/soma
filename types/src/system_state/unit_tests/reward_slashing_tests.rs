// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! C2 regression tests for the validator reward-slashing redistribution
//! math in [`ValidatorSet::compute_adjusted_reward_distribution`].

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod reward_slashing_tests {
    use std::collections::BTreeMap;

    use crate::base::dbg_addr;
    use crate::config::genesis_config::SHANNONS_PER_SOMA;
    use crate::system_state::test_utils::create_validator_for_testing;
    use crate::system_state::validator::ValidatorSet;

    /// The slashing-bonus redistribution must not divide by zero when there
    /// is no unslashed voting power to redistribute the slashed bonus to.
    ///
    /// Degenerate setup: validator 0 holds all the voting power and is
    /// slashed; validator 1 is unslashed but has zero voting power. Then
    /// `total_unslashed_voting_power == 0`, and the pre-fix bonus
    /// computation `total_adjustment * power / 0` divided by zero — a panic
    /// inside `advance_epoch` that safe mode does NOT catch (it only
    /// converts `Result::Err`, not panics) and that `panic = 'abort'` turns
    /// into a fleet-wide node crash.
    ///
    /// Before the fix this panicked; after, the zero-power unslashed
    /// validator simply receives no bonus.
    #[test]
    fn adjusted_reward_distribution_guards_zero_unslashed_power() {
        let mut validators = vec![
            create_validator_for_testing(dbg_addr(1), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(dbg_addr(2), 100 * SHANNONS_PER_SOMA),
        ];
        validators[0].voting_power = 10_000; // will be slashed
        validators[1].voting_power = 0; // unslashed, zero power
        let set = ValidatorSet::new(validators);

        let unadjusted = vec![1_000u64, 0u64];
        let total_adjustment = 500u64;
        let mut individual = BTreeMap::new();
        individual.insert(0usize, 500u64); // validator 0 is slashed

        // total_voting_power = 10_000, total_slashed = 10_000 → unslashed = 0.
        let adjusted = set.compute_adjusted_reward_distribution(
            10_000,
            10_000,
            unadjusted,
            total_adjustment,
            &individual,
        );

        assert_eq!(adjusted.len(), 2);
        assert_eq!(adjusted[0], 500, "slashed validator: 1000 reward - 500 adjustment");
        assert_eq!(
            adjusted[1], 0,
            "zero-power unslashed validator receives no bonus (no divide-by-zero)"
        );
    }

    /// Sanity: in the normal case (some unslashed power), the slashed
    /// bonus redistributes proportionally to unslashed voting power.
    #[test]
    fn adjusted_reward_distribution_redistributes_bonus_when_unslashed_power_exists() {
        let mut validators = vec![
            create_validator_for_testing(dbg_addr(1), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(dbg_addr(2), 100 * SHANNONS_PER_SOMA),
        ];
        validators[0].voting_power = 5_000; // slashed
        validators[1].voting_power = 5_000; // unslashed
        let set = ValidatorSet::new(validators);

        let unadjusted = vec![1_000u64, 1_000u64];
        let total_adjustment = 400u64; // validator 0's slashed amount
        let mut individual = BTreeMap::new();
        individual.insert(0usize, 400u64);

        let adjusted = set.compute_adjusted_reward_distribution(
            10_000,
            5_000, // total_slashed (validator 0)
            unadjusted,
            total_adjustment,
            &individual,
        );

        assert_eq!(adjusted[0], 600, "slashed: 1000 - 400");
        // bonus = 400 * 5000 / 5000 = 400 → unslashed gets full redistribution.
        assert_eq!(adjusted[1], 1_400, "unslashed: 1000 + 400 bonus");
    }
}
