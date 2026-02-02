#[cfg(test)]
mod delegation_tests {
    use crate::{
        base::{SomaAddress, dbg_addr},
        config::genesis_config::SHANNONS_PER_SOMA,
        effects::ExecutionFailureStatus,
        error::SomaError,
        system_state::{
            SystemParameters, SystemState,
            staking::StakedSoma,
            test_utils::{
                self, add_validator, advance_epoch_with_reward_amounts, advance_epoch_with_rewards,
                assert_validator_total_stake_amounts, create_test_system_state,
                create_validator_for_testing, stake_with, total_soma_balance, unstake,
                validator_stake_amount,
            },
            validator::Validator,
        },
    };
    use std::collections::BTreeMap;

    // Create constant validator addresses for testing
    fn validator_addr_1() -> SomaAddress {
        dbg_addr(1)
    }
    fn validator_addr_2() -> SomaAddress {
        dbg_addr(2)
    }
    fn new_validator_addr() -> SomaAddress {
        dbg_addr(3)
    }

    // Create constant staker addresses for testing
    fn staker_addr_1() -> SomaAddress {
        dbg_addr(4)
    }
    fn staker_addr_2() -> SomaAddress {
        dbg_addr(5)
    }
    fn staker_addr_3() -> SomaAddress {
        dbg_addr(6)
    }

    // Helper to set up a test system state
    fn set_up_system_state() -> SystemState {
        let validators = vec![
            create_validator_for_testing(validator_addr_1(), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr_2(), 100 * SHANNONS_PER_SOMA),
        ];

        create_test_system_state(validators, 1000, 0)
    }

    // Helper to set up a test system state with subsidy
    fn set_up_system_state_with_subsidy() -> SystemState {
        let validators = vec![
            create_validator_for_testing(validator_addr_1(), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr_2(), 100 * SHANNONS_PER_SOMA),
        ];

        create_test_system_state(validators, 400, 0)
    }

    // Helper to remove a validator candidate
    fn remove_validator_candidate(system_state: &mut SystemState, address: SomaAddress) {
        // Find and remove the validator from pending_active_validators
        let idx = system_state
            .validators
            .pending_validators
            .iter()
            .position(|v| v.metadata.soma_address == address)
            .expect("Validator candidate not found");

        system_state.validators.pending_validators.remove(idx);
    }

    #[test]
    fn test_add_remove_stake_flow() {
        let mut system_state = set_up_system_state();

        // Create a stake to VALIDATOR_ADDR_1
        let staked_soma = stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 60);

        // Verify initial stake amounts
        assert_eq!(
            validator_stake_amount(&system_state, validator_addr_1()).unwrap(),
            100 * SHANNONS_PER_SOMA
        );
        assert_eq!(
            validator_stake_amount(&system_state, validator_addr_2()).unwrap(),
            100 * SHANNONS_PER_SOMA
        );

        // Advance epoch to activate stake
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Verify stake has been added
        assert_eq!(
            validator_stake_amount(&system_state, validator_addr_1()).unwrap(),
            160 * SHANNONS_PER_SOMA
        );
        assert_eq!(
            validator_stake_amount(&system_state, validator_addr_2()).unwrap(),
            100 * SHANNONS_PER_SOMA
        );

        // Unstake from VALIDATOR_ADDR_1
        let unstaked_amount = unstake(&mut system_state, staked_soma);
        assert_eq!(unstaked_amount, 60 * SHANNONS_PER_SOMA);

        // Verify stake is still present until next epoch
        assert_eq!(
            validator_stake_amount(&system_state, validator_addr_1()).unwrap(),
            160 * SHANNONS_PER_SOMA
        );

        // Advance epoch to process withdrawal
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Verify stake has been removed
        assert_eq!(
            validator_stake_amount(&system_state, validator_addr_1()).unwrap(),
            100 * SHANNONS_PER_SOMA
        );
    }

    #[test]
    fn test_remove_stake_post_active_flow_no_rewards() {
        test_remove_stake_post_active_flow(false);
    }

    #[test]
    fn test_remove_stake_post_active_flow_with_rewards() {
        test_remove_stake_post_active_flow(true);
    }

    fn test_remove_stake_post_active_flow(should_distribute_rewards: bool) {
        let mut system_state = set_up_system_state_with_subsidy();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Add stake to validator 1
        let staked_soma = stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 100);

        // Advance epoch to activate stake
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Verify total stake amounts
        assert_validator_total_stake_amounts(
            &system_state,
            vec![validator_addr_1(), validator_addr_2()],
            vec![200 * SHANNONS_PER_SOMA, 100 * SHANNONS_PER_SOMA],
        );

        // Distribute rewards if needed
        if should_distribute_rewards {
            // Each validator pool gets significant rewards
            let _ = advance_epoch_with_rewards(&mut system_state, 80 * SHANNONS_PER_SOMA).unwrap();
        } else {
            let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();
        }

        // Request to remove validator 1
        let _ = system_state
            .request_remove_validator(validator_addr_1(), vec![])
            .unwrap();

        // Advance epoch to process validator removal
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Calculate expected reward amounts
        let reward_amt = if should_distribute_rewards {
            20 * SHANNONS_PER_SOMA
        } else {
            0
        };

        // Unstake from removed validator
        let withdrawn = unstake(&mut system_state, staked_soma);
        staker_withdrawals.insert(staker_addr_1(), withdrawn);

        // Verify withdrawn amount includes principal + rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()),
            100 * SHANNONS_PER_SOMA + reward_amt,
            "Staker should receive principal + rewards"
        );

        // Get validator's staked tokens
        let validator_pool_id = system_state
            .validators
            .staking_pool_mappings
            .iter()
            .find_map(|(id, addr)| {
                if *addr == validator_addr_1() {
                    Some(id)
                } else {
                    None
                }
            })
            .unwrap();

        // Find validator in inactive validators
        let validator = system_state
            .validators
            .inactive_validators
            .get(validator_pool_id)
            .unwrap();

        // Validator should still have their self-stake + rewards
        assert_eq!(
            validator.staking_pool.soma_balance,
            100 * SHANNONS_PER_SOMA + reward_amt,
            "Validator should retain self-stake + rewards"
        );
    }

    #[test]
    fn test_earns_rewards_at_last_epoch() {
        let mut system_state = set_up_system_state_with_subsidy();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Add stake to validator 1
        let staked_soma = stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 100);

        // Advance epoch to activate stake
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Request to remove validator 1
        let _ = system_state
            .request_remove_validator(validator_addr_1(), vec![])
            .unwrap();

        // Add rewards after the validator requests to leave
        // Since the validator is still active this epoch, it should get rewards
        let _ = advance_epoch_with_rewards(&mut system_state, 80 * SHANNONS_PER_SOMA).unwrap();

        // Expected rewards amounts
        let reward_amt = 20 * SHANNONS_PER_SOMA;

        // Unstake from removed validator
        let withdrawn = unstake(&mut system_state, staked_soma);
        staker_withdrawals.insert(staker_addr_1(), withdrawn);

        // Verify withdrawn amount includes principal + rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()),
            100 * SHANNONS_PER_SOMA + reward_amt,
            "Staker should receive principal + rewards from last epoch"
        );

        // Get validator's staked tokens
        let validator_pool_id = system_state
            .validators
            .staking_pool_mappings
            .iter()
            .find_map(|(id, addr)| {
                if *addr == validator_addr_1() {
                    Some(id)
                } else {
                    None
                }
            })
            .unwrap();

        // Find validator in inactive validators
        let validator = system_state
            .validators
            .inactive_validators
            .get(validator_pool_id)
            .unwrap();

        // Validator should have their self-stake + rewards
        assert_eq!(
            validator.staking_pool.soma_balance,
            100 * SHANNONS_PER_SOMA + reward_amt,
            "Validator should retain self-stake + rewards from last epoch"
        );
    }

    #[test]
    fn test_add_stake_post_active_flow() {
        let mut system_state = set_up_system_state();

        // Add stake to validator 1
        let staked_soma = stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 100);

        // Advance epoch to activate stake
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Request to remove validator 1
        let _ = system_state
            .request_remove_validator(validator_addr_1(), vec![])
            .unwrap();

        // Advance epoch to process validator removal
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Verify validator is no longer active
        assert!(
            !system_state
                .validators
                .is_active_validator(validator_addr_1())
        );

        // Try to add stake to the inactive validator - should fail
        let result = system_state.request_add_stake(
            staker_addr_1(),
            validator_addr_1(),
            60 * SHANNONS_PER_SOMA,
        );

        // Verify the error is as expected
        assert!(result.is_err());
        match result {
            Err(ExecutionFailureStatus::ValidatorNotFound) => {} // Expected error
            _ => panic!("Expected ValidatorNotFound error, got {:?}", result),
        }
    }

    #[test]
    fn test_add_preactive_remove_active() {
        let mut system_state = set_up_system_state_with_subsidy();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Advance epoch with rewards
        let _ = advance_epoch_with_rewards(&mut system_state, 300 * SHANNONS_PER_SOMA).unwrap();

        // Add a validator candidate
        add_validator(&mut system_state, new_validator_addr());

        // Delegate to the preactive validator
        let staked_soma_1 = stake_with(
            &mut system_state,
            staker_addr_1(),
            new_validator_addr(),
            100,
        );

        // Add more stakes
        let staked_soma_2 =
            stake_with(&mut system_state, staker_addr_2(), new_validator_addr(), 50);
        let staked_soma_3 = stake_with(
            &mut system_state,
            staker_addr_3(),
            new_validator_addr(),
            100,
        );

        // Advance epoch to activate validator
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Distribute rewards
        let _ = advance_epoch_with_rewards(&mut system_state, 85 * SHANNONS_PER_SOMA).unwrap();

        // Stakers 1 and 3 unstake and should earn proportional rewards
        let withdrawn_1 = unstake(&mut system_state, staked_soma_1);
        staker_withdrawals.insert(staker_addr_1(), withdrawn_1);

        let withdrawn_3 = unstake(&mut system_state, staked_soma_3);
        staker_withdrawals.insert(staker_addr_3(), withdrawn_3);

        // Both should get similar rewards
        let staker_1_bal = total_soma_balance(&staker_withdrawals, staker_addr_1());
        let staker_3_bal = total_soma_balance(&staker_withdrawals, staker_addr_3());

        // They earn the same rewards as long as they unstake
        // in the same epoch because the validator was preactive when they staked.
        // So they will both get slightly more than 110 SOMA in total balance.
        assert_eq!(staker_1_bal, 111332200000);
        assert_eq!(staker_3_bal, 111332200000);

        // Distribute more rewards
        let _ = advance_epoch_with_rewards(&mut system_state, 85 * SHANNONS_PER_SOMA).unwrap();

        // Staker 2 unstakes and should get additional rewards
        let withdrawn_2 = unstake(&mut system_state, staked_soma_2);
        staker_withdrawals.insert(staker_addr_2(), withdrawn_2);

        let staker_2_bal = total_soma_balance(&staker_withdrawals, staker_addr_2());

        assert_eq!(staker_2_bal, 83996600000);
    }

    #[test]
    fn test_add_preactive_remove_post_active() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Add a validator candidate

        add_validator(&mut system_state, new_validator_addr());

        // Delegate to the preactive validator
        let staked_soma = stake_with(
            &mut system_state,
            staker_addr_1(),
            new_validator_addr(),
            100,
        );

        // Advance epoch to activate validator
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Distribute rewards
        let _ = advance_epoch_with_rewards(&mut system_state, 90 * SHANNONS_PER_SOMA).unwrap();

        // Remove validator
        let _ = system_state
            .request_remove_validator(new_validator_addr(), vec![])
            .unwrap();

        // Advance epoch to process validator removal
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Unstake and check rewards
        let withdrawn = unstake(&mut system_state, staked_soma);
        staker_withdrawals.insert(staker_addr_1(), withdrawn);

        // Staker should get around 130 SOMA (100 principal + 30 rewards)
        let staker_bal = total_soma_balance(&staker_withdrawals, staker_addr_1());
        assert_eq!(staker_bal, 129997000000);
    }

    #[test]
    fn test_staking_pool_exchange_rate_getter() {
        let mut system_state = set_up_system_state();

        // Stake with validator 2
        let staked_soma = stake_with(&mut system_state, staker_addr_1(), validator_addr_2(), 100);
        let pool_id = staked_soma.pool_id;

        // Advance epoch to activate stake
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Distribute rewards
        let _ = advance_epoch_with_rewards(&mut system_state, 20 * SHANNONS_PER_SOMA).unwrap();

        // Check exchange rates
        let validator = system_state
            .validators
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == validator_addr_2())
            .expect("Validator not found");

        // Check exchange rates for each epoch
        let rate_0 = validator.pool_token_exchange_rate_at_epoch(0);
        assert_eq!(rate_0.soma_amount, 100 * SHANNONS_PER_SOMA); // Initial stake at genesis
        assert_eq!(rate_0.pool_token_amount, 100 * SHANNONS_PER_SOMA);

        let rate_1 = validator.pool_token_exchange_rate_at_epoch(1);
        assert_eq!(rate_1.soma_amount, 200 * SHANNONS_PER_SOMA); // 100 self + 100 delegated
        assert_eq!(rate_1.pool_token_amount, 200 * SHANNONS_PER_SOMA);

        let rate_2 = validator.pool_token_exchange_rate_at_epoch(2);
        assert_eq!(rate_2.soma_amount, 210 * SHANNONS_PER_SOMA); // 200 + 10 rewards
        assert_eq!(rate_2.pool_token_amount, 200 * SHANNONS_PER_SOMA);
    }
}
