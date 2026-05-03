// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0
//
// F1 delegation lifecycle tests at the SystemState API level.
//
// The pre-Stage-9d versions of these tests asserted compounded
// per-staker amounts that flowed through the pool-token /
// exchange-rate machinery. Under F1 the per-staker delegation row
// lives in the authority crate's `delegations` column family — the
// types crate only owns the pool-aggregate side. So these tests
// cover:
//
//   1. Stake added to a validator increases that pool's
//      `total_stake` (drives voting power).
//   2. Stake withdrawn decreases `total_stake` symmetrically.
//   3. Validator removal moves the pool to `inactive_validators`,
//      preserving its accumulated state.
//   4. Stake additions to a removed/inactive validator are rejected.
//   5. Withdrawals from an inactive pool still work (a delegator can
//      always pull their principal back).
//
// Per-staker reward payout (F1 fold-to-balance) is exercised in
// `authority::unit_tests::staking_tests` because it depends on the
// delegations-table read path.

#[cfg(test)]
#[allow(clippy::module_inception, clippy::unwrap_used, clippy::expect_used)]
mod delegation_tests {
    use crate::base::{SomaAddress, dbg_addr};
    use crate::config::genesis_config::SHANNONS_PER_SOMA;
    use crate::effects::ExecutionFailureStatus;
    use crate::system_state::SystemState;
    use crate::system_state::test_utils::{
        ValidatorRewards, advance_epoch_with_reward_amounts, create_test_system_state,
        create_validator_for_testing,
    };

    fn validator_addr(seed: u8) -> SomaAddress {
        dbg_addr(seed)
    }

    /// 2-validator setup: 100 + 100 SOMA self-stake. 1000 SOMA in
    /// subsidy fund.
    fn set_up_2_validators() -> SystemState {
        let validators = vec![
            create_validator_for_testing(validator_addr(1), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr(2), 100 * SHANNONS_PER_SOMA),
        ];
        create_test_system_state(validators, 1000, 0)
    }

    fn validator_total_stake(state: &SystemState, addr: SomaAddress) -> Option<u64> {
        state
            .validators()
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == addr)
            .map(|v| v.staking_pool.total_stake)
    }

    fn inactive_pool_total_stake(state: &SystemState, addr: SomaAddress) -> Option<u64> {
        state
            .validators()
            .inactive_validators
            .values()
            .find(|v| v.metadata.soma_address == addr)
            .map(|v| v.staking_pool.total_stake)
    }

    /// Adding stake to an active validator immediately bumps its
    /// pool's `total_stake`. The bump persists across epochs.
    #[test]
    fn add_stake_increases_pool_total_stake() {
        let mut state = set_up_2_validators();
        assert_eq!(
            validator_total_stake(&state, validator_addr(1)),
            Some(100 * SHANNONS_PER_SOMA),
        );

        // Stake 60 SOMA into v1.
        let pool_id = state
            .add_stake_to_validator(validator_addr(1), 60 * SHANNONS_PER_SOMA)
            .expect("add_stake_to_validator");
        assert_eq!(
            validator_total_stake(&state, validator_addr(1)),
            Some(160 * SHANNONS_PER_SOMA),
            "total_stake bumps immediately",
        );
        // pool_id matches the validator's own staking_pool.id.
        assert_eq!(
            state.validators().validators[0].staking_pool.id,
            pool_id,
            "returned pool_id matches the validator's pool",
        );

        // v2 is untouched.
        assert_eq!(
            validator_total_stake(&state, validator_addr(2)),
            Some(100 * SHANNONS_PER_SOMA),
        );

        // Persists across an epoch boundary.
        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        assert_eq!(
            validator_total_stake(&state, validator_addr(1)),
            Some(160 * SHANNONS_PER_SOMA),
        );
    }

    /// Removing principal symmetrically drops `total_stake`. A full
    /// drain leaves the pool at 0 — but the pool still exists (with
    /// the validator's own genesis self-stake as the floor, since
    /// the validator's row was the genesis seed).
    #[test]
    fn remove_stake_decreases_pool_total_stake() {
        let mut state = set_up_2_validators();
        let pool_id = state
            .add_stake_to_validator(validator_addr(1), 60 * SHANNONS_PER_SOMA)
            .expect("add");
        assert_eq!(
            validator_total_stake(&state, validator_addr(1)),
            Some(160 * SHANNONS_PER_SOMA),
        );

        state
            .remove_stake_from_validator(pool_id, 60 * SHANNONS_PER_SOMA)
            .expect("remove");
        assert_eq!(
            validator_total_stake(&state, validator_addr(1)),
            Some(100 * SHANNONS_PER_SOMA),
            "removing the same amount returns to baseline",
        );
    }

    /// Zero stake amount is rejected at `add_stake_to_validator`.
    #[test]
    fn zero_stake_amount_rejected() {
        let mut state = set_up_2_validators();
        let err = state
            .add_stake_to_validator(validator_addr(1), 0)
            .expect_err("zero stake must error");
        match err {
            ExecutionFailureStatus::InvalidArguments { .. } => {}
            other => panic!("expected InvalidArguments, got {:?}", other),
        }
    }

    /// Stake to an unknown validator address is rejected.
    #[test]
    fn unknown_validator_rejected() {
        let mut state = set_up_2_validators();
        let unknown = dbg_addr(99);
        let err = state
            .add_stake_to_validator(unknown, 50 * SHANNONS_PER_SOMA)
            .expect_err("unknown validator must error");
        match err {
            ExecutionFailureStatus::ValidatorNotFound => {}
            other => panic!("expected ValidatorNotFound, got {:?}", other),
        }
    }

    /// After a validator is removed and the epoch transitions, the
    /// validator's pool moves to `inactive_validators` with its
    /// `total_stake` preserved. Stake can still be withdrawn from
    /// the inactive pool.
    #[test]
    fn removed_validator_pool_persists_in_inactive_set() {
        let mut state = set_up_2_validators();

        // Add a delegator's stake.
        let pool_id = state
            .add_stake_to_validator(validator_addr(1), 50 * SHANNONS_PER_SOMA)
            .expect("add");
        let total_before_removal = validator_total_stake(&state, validator_addr(1)).unwrap();
        assert_eq!(total_before_removal, 150 * SHANNONS_PER_SOMA);

        // Request validator removal.
        state.validators_mut().request_remove_validator(validator_addr(1)).expect("remove");

        // Run an epoch boundary to process the removal.
        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);

        // No longer active.
        assert!(
            !state.validators().is_active_validator(validator_addr(1)),
            "v1 must no longer be active",
        );
        // total_stake preserved on the inactive pool.
        assert_eq!(
            inactive_pool_total_stake(&state, validator_addr(1)),
            Some(total_before_removal),
            "inactive pool retains its total_stake",
        );

        // Withdrawal from the inactive pool still works.
        state
            .remove_stake_from_validator(pool_id, 50 * SHANNONS_PER_SOMA)
            .expect("remove from inactive");
        assert_eq!(
            inactive_pool_total_stake(&state, validator_addr(1)),
            Some(100 * SHANNONS_PER_SOMA),
            "inactive pool's total_stake drops on withdrawal",
        );
    }

    /// Once a validator has been removed and the epoch boundary has
    /// processed the removal, AddStake to that validator's *address*
    /// is rejected — the address is no longer in the active or
    /// pending sets, only the inactive map (which is keyed by
    /// pool_id, not validator_address).
    #[test]
    fn add_stake_to_removed_validator_rejected() {
        let mut state = set_up_2_validators();

        state.validators_mut().request_remove_validator(validator_addr(1)).expect("remove");

        let mut tracker = ValidatorRewards::new(&state.validators().validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);

        let err = state
            .add_stake_to_validator(validator_addr(1), 60 * SHANNONS_PER_SOMA)
            .expect_err("add to removed validator must error");
        match err {
            ExecutionFailureStatus::ValidatorNotFound => {}
            other => panic!("expected ValidatorNotFound, got {:?}", other),
        }
    }

    /// `staking_pool_mappings` is updated whenever a stake-set change
    /// hits a validator. Before any activity, mappings already carry
    /// the genesis validators' pool ids; after `add_stake_to_validator`
    /// the same pool id is reachable via lookup. Important for
    /// `remove_stake_from_validator` which routes by pool_id.
    #[test]
    fn staking_pool_mappings_resolves_pool_id_to_validator() {
        let mut state = set_up_2_validators();
        let pool_id =
            state.add_stake_to_validator(validator_addr(1), 1).expect("add");
        let mapped = state
            .validators()
            .staking_pool_mappings
            .get(&pool_id)
            .copied()
            .expect("mapping must exist");
        assert_eq!(mapped, validator_addr(1));
    }

    /// Withdraw via an unknown pool_id is rejected.
    #[test]
    fn unknown_pool_id_rejected() {
        let mut state = set_up_2_validators();
        let unknown_pool = crate::object::ObjectID::random();
        let err = state
            .remove_stake_from_validator(unknown_pool, 1)
            .expect_err("unknown pool must error");
        match err {
            ExecutionFailureStatus::StakingPoolNotFound => {}
            other => panic!("expected StakingPoolNotFound, got {:?}", other),
        }
    }

    /// `add_stake_to_validator` is idempotent across epochs — a
    /// stake added in epoch 0 and another in epoch 1 sum into the
    /// same pool's `total_stake`.
    #[test]
    fn repeat_stake_additions_accumulate() {
        let mut state = set_up_2_validators();
        let mut tracker = ValidatorRewards::new(&state.validators().validators);

        state.add_stake_to_validator(validator_addr(1), 25 * SHANNONS_PER_SOMA).expect("add 1");
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);
        state.add_stake_to_validator(validator_addr(1), 35 * SHANNONS_PER_SOMA).expect("add 2");
        advance_epoch_with_reward_amounts(&mut state, 0, &mut tracker);

        assert_eq!(
            validator_total_stake(&state, validator_addr(1)),
            Some((100 + 25 + 35) * SHANNONS_PER_SOMA),
            "successive stake additions accumulate into total_stake",
        );
    }
}
