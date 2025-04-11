#[cfg(test)]
mod rewards_distribution_tests {
    use crate::{
        base::{dbg_addr, SomaAddress},
        system_state::{
            test_utils::{
                self, advance_epoch_with_reward_amounts,
                advance_epoch_with_reward_amounts_and_slashing_rates,
                assert_validator_non_self_stake_amounts, assert_validator_self_stake_amounts,
                assert_validator_total_stake_amounts, stake_with, total_soma_balance, unstake,
                ValidatorRewards,
            },
            validator::Validator,
            SystemState,
        },
    };
    use std::collections::BTreeMap;

    // Constants for testing
    const SHANNONS_PER_SOMA: u64 = 1_000_000_000;

    // Create constant validator addresses for testing
    fn validator_addr_1() -> SomaAddress {
        dbg_addr(1)
    }
    fn validator_addr_2() -> SomaAddress {
        dbg_addr(2)
    }
    fn validator_addr_3() -> SomaAddress {
        dbg_addr(3)
    }
    fn validator_addr_4() -> SomaAddress {
        dbg_addr(4)
    }

    // Create constant staker addresses for testing
    fn staker_addr_1() -> SomaAddress {
        dbg_addr(5)
    }
    fn staker_addr_2() -> SomaAddress {
        dbg_addr(6)
    }
    fn staker_addr_3() -> SomaAddress {
        dbg_addr(7)
    }
    fn staker_addr_4() -> SomaAddress {
        dbg_addr(8)
    }

    #[test]
    fn test_validator_rewards() {
        let mut system_state = set_up_system_state();
        // Record initial validator states
        let mut validator_stakes =
            ValidatorRewards::new(&system_state.validators.active_validators);

        // Need to advance epoch so validator's staking starts counting
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Each validator gets 25 SOMA (total 100 SOMA rewards)
        advance_epoch_with_reward_amounts(&mut system_state, 100, &mut validator_stakes);

        // Validator total stake should increase by their share of rewards
        assert_validator_total_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                125 * SHANNONS_PER_SOMA,
                225 * SHANNONS_PER_SOMA,
                325 * SHANNONS_PER_SOMA,
                425 * SHANNONS_PER_SOMA,
            ],
        );

        // Add a lot more stake to validator 2 to test voting power cap
        stake_with(
            &mut system_state,
            validator_addr_2(),
            validator_addr_2(),
            720,
        );

        // Advance epoch to activate new stake
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Distribute more rewards (100 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 100, &mut validator_stakes);

        // Even though validator 2 has more stake, rewards should respect voting power cap
        assert_validator_total_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                150 * SHANNONS_PER_SOMA,
                970 * SHANNONS_PER_SOMA,
                350 * SHANNONS_PER_SOMA,
                450 * SHANNONS_PER_SOMA,
            ],
        );
    }

    #[test]
    fn test_stake_subsidy() {
        let mut system_state = set_up_system_state_with_big_amounts();

        // Record initial validator states
        let mut validator_stakes =
            ValidatorRewards::new(&system_state.validators.active_validators);

        // Need to advance epoch so validator's staking starts counting
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Each validator gets 25 SOMA (total 100 SOMA rewards)
        advance_epoch_with_reward_amounts(&mut system_state, 100, &mut validator_stakes);

        // Validator total stake should increase by their share of rewards
        assert_validator_total_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                100_000_025 * SHANNONS_PER_SOMA,
                200_000_025 * SHANNONS_PER_SOMA,
                300_000_025 * SHANNONS_PER_SOMA,
                400_000_025 * SHANNONS_PER_SOMA,
            ],
        );
    }

    #[test]
    fn test_stake_rewards() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Record initial validator states
        let mut validator_stakes =
            ValidatorRewards::new(&system_state.validators.active_validators);

        // Add stake to validators
        let staked_soma_1 = stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 200);
        let staked_soma_2 = stake_with(&mut system_state, staker_addr_2(), validator_addr_2(), 100);

        // Advance epoch to activate stake
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Verify total stake amounts
        assert_validator_total_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                300 * SHANNONS_PER_SOMA,
                300 * SHANNONS_PER_SOMA,
                300 * SHANNONS_PER_SOMA,
                400 * SHANNONS_PER_SOMA,
            ],
        );

        // Verify validator self-stake amounts - just initial stakes at this point
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                100 * SHANNONS_PER_SOMA,
                200 * SHANNONS_PER_SOMA,
                300 * SHANNONS_PER_SOMA,
                400 * SHANNONS_PER_SOMA,
            ],
            &validator_stakes,
        );

        // Each validator gets 30 SOMA (total 120 SOMA rewards)
        advance_epoch_with_reward_amounts(&mut system_state, 120, &mut validator_stakes);

        // Verify validator self-stake amounts after rewards
        // Self-stake should grow proportionally
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                110 * SHANNONS_PER_SOMA,
                220 * SHANNONS_PER_SOMA,
                330 * SHANNONS_PER_SOMA,
                430 * SHANNONS_PER_SOMA,
            ],
            &validator_stakes,
        );

        // Unstake and track withdrawal amount
        let withdrawn_1 = unstake(&mut system_state, staked_soma_1);
        staker_withdrawals.insert(staker_addr_1(), withdrawn_1);

        // Add more stake to validator 1
        let staked_soma_3 = stake_with(&mut system_state, staker_addr_2(), validator_addr_1(), 600);

        // Distribute more rewards (120 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 120, &mut validator_stakes);

        // Staker 1 should have received rewards proportional to their stake
        // Verify the withdrawal includes principal + rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()),
            220 * SHANNONS_PER_SOMA
        );

        // Validator self-stake amounts should increase with their share of rewards
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                140 * SHANNONS_PER_SOMA,
                240 * SHANNONS_PER_SOMA,
                360 * SHANNONS_PER_SOMA,
                460 * SHANNONS_PER_SOMA,
            ],
            &validator_stakes,
        );

        // Unstake staker 2's first stake and track withdrawal
        let withdrawn_2 = unstake(&mut system_state, staked_soma_2);
        staker_withdrawals.insert(staker_addr_2(), withdrawn_2);

        // Verify staker 2's first withdrawal includes rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_2()),
            120 * SHANNONS_PER_SOMA
        );

        // Distribute more rewards (40 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 40, &mut validator_stakes);

        // Unstake staker 2's second stake and add to tracked withdrawals
        let withdrawn_3 = unstake(&mut system_state, staked_soma_3);
        staker_withdrawals.insert(
            staker_addr_2(),
            total_soma_balance(&staker_withdrawals, staker_addr_2()) + withdrawn_3,
        );

        // Verify total withdrawal amount for staker 2
        let staker_2_balance = total_soma_balance(&staker_withdrawals, staker_addr_2());
        assert_eq!(staker_2_balance, 728108108107);
    }

    #[test]
    fn test_stake_tiny_rewards() {
        let mut system_state = set_up_system_state_with_big_amounts();

        // Record initial validator states
        let mut validator_stakes =
            ValidatorRewards::new(&system_state.validators.active_validators);

        // Stake a large amount
        let staked_soma_1 = stake_with(
            &mut system_state,
            staker_addr_1(),
            validator_addr_1(),
            200_000_000,
        );

        // Advance epoch to activate stake
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Distribute significant rewards
        advance_epoch_with_reward_amounts(&mut system_state, 150_000, &mut validator_stakes);

        // Stake a small amount to the same validator
        let staked_soma_2 = stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 10);

        // Distribute small rewards
        advance_epoch_with_reward_amounts(&mut system_state, 130, &mut validator_stakes);

        // Unstake the small stake
        unstake(&mut system_state, staked_soma_2);

        // Distribute more rewards and ensure no errors
        advance_epoch_with_reward_amounts(&mut system_state, 150, &mut validator_stakes);

        // Unstake the large stake and verify it succeeded
        let withdrawn = unstake(&mut system_state, staked_soma_1);
        assert!(
            withdrawn > 200_000_000 * SHANNONS_PER_SOMA,
            "Should have received rewards"
        );
    }

    #[test]
    fn test_validator_commission() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Record initial validator states
        let mut validator_stakes =
            ValidatorRewards::new(&system_state.validators.active_validators);

        // Add stake to validators
        let staked_soma_1 = stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 100);
        let staked_soma_2 = stake_with(&mut system_state, staker_addr_2(), validator_addr_2(), 100);

        // Advance epoch to activate stake
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Set commission rate for validator 2 to 20% (2000 basis points)
        set_commission_rate(&mut system_state, validator_addr_2(), 2000);

        // Advance epoch to apply commission rate change
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Distribute rewards (120 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 120, &mut validator_stakes);

        // Check non-self stake amounts
        assert_validator_non_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![115 * SHANNONS_PER_SOMA, 108 * SHANNONS_PER_SOMA, 0, 0],
            &validator_stakes,
        );

        // Check validator self stake amounts
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                115 * SHANNONS_PER_SOMA,
                222 * SHANNONS_PER_SOMA,
                330 * SHANNONS_PER_SOMA,
                430 * SHANNONS_PER_SOMA,
            ],
            &validator_stakes,
        );

        // Set commission rate for validator 1 to 10% (1000 basis points)
        set_commission_rate(&mut system_state, validator_addr_1(), 1000);

        // Advance epoch to apply commission rate change
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Distribute more rewards (240 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 240, &mut validator_stakes);

        // Verify total stake amounts
        assert_validator_total_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                290 * SHANNONS_PER_SOMA,
                390 * SHANNONS_PER_SOMA,
                390 * SHANNONS_PER_SOMA,
                490 * SHANNONS_PER_SOMA,
            ],
        );

        // Verify split between validator and staker stakes
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                148 * SHANNONS_PER_SOMA,
                266290909091,
                390 * SHANNONS_PER_SOMA,
                490 * SHANNONS_PER_SOMA,
            ],
            &validator_stakes,
        );

        assert_validator_non_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![142 * SHANNONS_PER_SOMA, 123709090909, 0, 0],
            &validator_stakes,
        );
    }

    #[test]
    fn test_rewards_slashing() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Record initial validator states
        let mut validator_stakes =
            ValidatorRewards::new(&system_state.validators.active_validators);

        // Advance epoch to start reward counting
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Add stake to validators
        let staked_soma_1 = stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 100);
        let staked_soma_2 = stake_with(&mut system_state, staker_addr_2(), validator_addr_2(), 100);

        // Advance epoch to activate stake
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Report validator 2 by 3 other validators (75% of stake)
        report_validator(&mut system_state, validator_addr_1(), validator_addr_2());
        report_validator(&mut system_state, validator_addr_3(), validator_addr_2());
        report_validator(&mut system_state, validator_addr_4(), validator_addr_2());

        // Report validator 1 by only 1 other validator (25% of stake)
        report_validator(&mut system_state, validator_addr_3(), validator_addr_1());

        // Distribute rewards (3600 SOMA) with 10% reward slashing for reported validators
        // In our implementation, advance_epoch has a built-in slashing mechanism
        advance_epoch_with_reward_amounts_and_slashing_rates(
            &mut system_state,
            3600,
            1000,
            &mut validator_stakes,
        );

        // Validator 2 should have 10% of rewards slashed
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                565 * SHANNONS_PER_SOMA,
                740 * SHANNONS_PER_SOMA,
                1230 * SHANNONS_PER_SOMA,
                1330 * SHANNONS_PER_SOMA,
            ],
            &validator_stakes,
        );

        // Unstake to check rewards
        let withdrawn_1 = unstake(&mut system_state, staked_soma_1);
        let withdrawn_2 = unstake(&mut system_state, staked_soma_2);

        staker_withdrawals.insert(staker_addr_1(), withdrawn_1);
        staker_withdrawals.insert(staker_addr_2(), withdrawn_2);

        // Staker 1 should get full rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()),
            565 * SHANNONS_PER_SOMA
        );

        // Staker 2 should have slashed rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_2()),
            370 * SHANNONS_PER_SOMA
        );
    }

    #[test]
    fn test_entire_rewards_slashing() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Record initial validator states
        let mut validator_stakes =
            ValidatorRewards::new(&system_state.validators.active_validators);

        // Advance epoch to start reward counting
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Add stake to validators
        let staked_soma_1 = stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 100);
        let staked_soma_2 = stake_with(&mut system_state, staker_addr_2(), validator_addr_2(), 100);

        // Advance epoch to activate stake
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Report validator 2 by 3 other validators (75% of stake)
        report_validator(&mut system_state, validator_addr_1(), validator_addr_2());
        report_validator(&mut system_state, validator_addr_3(), validator_addr_2());
        report_validator(&mut system_state, validator_addr_4(), validator_addr_2());

        // Distribute rewards (3600 SOMA) with 100% reward slashing
        advance_epoch_with_reward_amounts_and_slashing_rates(
            &mut system_state,
            3600,
            10000,
            &mut validator_stakes,
        );

        // Validator 2 should have all rewards slashed
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                700 * SHANNONS_PER_SOMA,
                200 * SHANNONS_PER_SOMA,
                1500 * SHANNONS_PER_SOMA,
                1600 * SHANNONS_PER_SOMA,
            ],
            &validator_stakes,
        );

        // Unstake to check rewards
        let withdrawn_1 = unstake(&mut system_state, staked_soma_1);
        let withdrawn_2 = unstake(&mut system_state, staked_soma_2);

        staker_withdrawals.insert(staker_addr_1(), withdrawn_1);
        staker_withdrawals.insert(staker_addr_2(), withdrawn_2);

        // Staker 1 should get additional rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()),
            700 * SHANNONS_PER_SOMA
        );

        // Staker 2 should only get principal back with no rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_2()),
            100 * SHANNONS_PER_SOMA
        );
    }

    #[test]
    fn test_mul_rewards_withdraws_at_same_epoch() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Record initial validator states
        let mut validator_stakes =
            ValidatorRewards::new(&system_state.validators.active_validators);

        // Add stake to validator 1
        let staked_soma_1_1 =
            stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 220);

        // Distribute rewards (40 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 40, &mut validator_stakes);

        // Add more stake
        let staked_soma_2_1 =
            stake_with(&mut system_state, staker_addr_2(), validator_addr_1(), 480);

        // Distribute rewards (120 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 120, &mut validator_stakes);

        // Add more stakes
        let staked_soma_1_2 =
            stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 130);
        let staked_soma_3_1 =
            stake_with(&mut system_state, staker_addr_3(), validator_addr_1(), 390);

        // Distribute rewards (280 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 280, &mut validator_stakes);

        // Add more stakes
        let staked_soma_3_2 =
            stake_with(&mut system_state, staker_addr_3(), validator_addr_1(), 280);
        let staked_soma_4_1 =
            stake_with(&mut system_state, staker_addr_4(), validator_addr_1(), 1400);

        // Distribute rewards (440 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 440, &mut validator_stakes);

        // Verify total stake in validator 1's pool
        let validator_1 = system_state
            .validators
            .active_validators
            .iter()
            .find(|v| v.metadata.soma_address == validator_addr_1())
            .expect("Validator 1 not found");

        assert_eq!(
            validator_1.staking_pool.soma_balance,
            140 * 23 * SHANNONS_PER_SOMA,
            "Expected validator 1 to have 140 * 23 SOMA"
        );

        // Withdraw all stakes at once
        let withdrawals = vec![
            (staker_addr_1(), unstake(&mut system_state, staked_soma_1_1)),
            (staker_addr_1(), unstake(&mut system_state, staked_soma_1_2)),
            (staker_addr_2(), unstake(&mut system_state, staked_soma_2_1)),
            (staker_addr_3(), unstake(&mut system_state, staked_soma_3_1)),
            (staker_addr_3(), unstake(&mut system_state, staked_soma_3_2)),
            (staker_addr_4(), unstake(&mut system_state, staked_soma_4_1)),
        ];

        // Process withdrawals
        for (addr, amount) in withdrawals {
            staker_withdrawals.insert(addr, total_soma_balance(&staker_withdrawals, addr) + amount);
        }

        // Verify staker balances after withdrawals
        // Staker 1's first stake was active for 3 epochs (60 SOMA rewards)
        // and second stake active for 1 epoch (10 SOMA rewards)
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()) / SHANNONS_PER_SOMA,
            220 + 130 + 20 * 3 + 10,
            "Incorrect withdrawal amount for staker 1"
        );

        // Staker 2's stake was active for 2 epochs (80 SOMA rewards)
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_2()) / SHANNONS_PER_SOMA,
            480 + 40 * 2,
            "Incorrect withdrawal amount for staker 2"
        );

        // Staker 3's first stake was active for 1 epoch (30 SOMA rewards)
        // and second stake had no rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_3()) / SHANNONS_PER_SOMA,
            390 + 280 + 30,
            "Incorrect withdrawal amount for staker 3"
        );

        // Staker 4 joined and left in the same epoch, so no rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_4()),
            1400 * SHANNONS_PER_SOMA,
            "Incorrect withdrawal amount for staker 4"
        );

        // Advance epoch one more time
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Verify validator pool now only has the validator's original stake
        let validator_1 = system_state
            .validators
            .active_validators
            .iter()
            .find(|v| v.metadata.soma_address == validator_addr_1())
            .expect("Validator 1 not found");

        assert_eq!(
            validator_1.staking_pool.soma_balance,
            140 * SHANNONS_PER_SOMA,
            "Expected validator 1 to have 140 SOMA after all withdrawals"
        );
    }

    #[test]
    fn test_uncapped_rewards() {
        // Create 20 validators with increasing stake
        let mut validators = Vec::new();
        let num_validators = 20;

        // The stake total sums up to 10000 SOMA
        for i in 0..num_validators {
            let addr = dbg_addr(i as u8 + 1);
            let stake = 481 + i * 2;
            validators.push(create_validator_for_testing(addr, stake));
        }

        // Create system state with these validators
        let mut system_state = test_utils::create_test_system_state(validators, 0, 0, 10, 0);

        // Record initial validator states
        let mut validator_stakes =
            ValidatorRewards::new(&system_state.validators.active_validators);

        // Double the stake of each validator through rewards
        advance_epoch_with_reward_amounts(&mut system_state, 10000, &mut validator_stakes);

        // Verify each validator's stake doubled
        for i in 0..num_validators {
            let addr = dbg_addr(i as u8 + 1);

            let validator = system_state
                .validators
                .active_validators
                .iter()
                .find(|v| v.metadata.soma_address == addr)
                .expect("Validator not found");

            let actual_stake = validator.staking_pool.soma_balance;
            let expected_stake = (962 + i * 4) * SHANNONS_PER_SOMA;

            assert_eq!(actual_stake, expected_stake);
        }
    }

    // HELPERS

    // Helper function to create test validators
    fn create_validator_for_testing(addr: SomaAddress, init_stake_amount: u64) -> Validator {
        // Use helper from our test_utils
        let mut validator =
            test_utils::create_validator_for_testing(addr, init_stake_amount * SHANNONS_PER_SOMA);

        // For these tests, activate the validator immediately
        if init_stake_amount > 0 {
            validator.staking_pool.activation_epoch = Some(0);
        }

        validator
    }

    // Helper function to create a system state with standard validators
    fn set_up_system_state() -> SystemState {
        let validators = vec![
            create_validator_for_testing(validator_addr_1(), 100),
            create_validator_for_testing(validator_addr_2(), 200),
            create_validator_for_testing(validator_addr_3(), 300),
            create_validator_for_testing(validator_addr_4(), 400),
        ];

        test_utils::create_test_system_state(
            validators, 1000, // 1000 SOMA subsidy fund
            0,    // 0 SOMA initial distribution (we'll set this per test)
            10,   // period length of 10 epochs
            500,  // 5% decrease rate
        )
    }

    // Helper function to create a system state with big amounts
    fn set_up_system_state_with_big_amounts() -> SystemState {
        let validators = vec![
            create_validator_for_testing(validator_addr_1(), 100_000_000),
            create_validator_for_testing(validator_addr_2(), 200_000_000),
            create_validator_for_testing(validator_addr_3(), 300_000_000),
            create_validator_for_testing(validator_addr_4(), 400_000_000),
        ];

        test_utils::create_test_system_state(
            validators,
            1_000_000_000, // 1B SOMA subsidy fund
            0,             // 0 SOMA initial distribution (we'll set this per test)
            10,            // period length of 10 epochs
            500,           // 5% decrease rate
        )
    }

    // Helper to get vector of validator addresses
    fn validator_addrs() -> Vec<SomaAddress> {
        vec![
            validator_addr_1(),
            validator_addr_2(),
            validator_addr_3(),
            validator_addr_4(),
        ]
    }

    // Helper to report a validator
    fn report_validator(
        system_state: &mut SystemState,
        reporter: SomaAddress,
        reportee: SomaAddress,
    ) {
        system_state
            .report_validator(reporter, reportee)
            .expect("Failed to report validator");
    }

    // Helper to set validator commission rate
    fn set_commission_rate(
        system_state: &mut SystemState,
        validator: SomaAddress,
        commission_rate: u64,
    ) {
        system_state
            .request_set_commission_rate(validator, commission_rate)
            .expect("Failed to set commission rate");
    }
}
