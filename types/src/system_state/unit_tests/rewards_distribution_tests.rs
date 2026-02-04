#[cfg(test)]
mod rewards_distribution_tests {
    use crate::{
        base::{SomaAddress, dbg_addr},
        config::genesis_config::SHANNONS_PER_SOMA,
        system_state::{
            SystemState,
            test_utils::{
                self, ValidatorRewards, advance_epoch_with_reward_amounts,
                advance_epoch_with_reward_amounts_and_slashing_rates,
                assert_validator_non_self_stake_amounts, assert_validator_self_stake_amounts,
                assert_validator_total_stake_amounts, stake_with, total_soma_balance, unstake,
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
        let mut validator_stakes = ValidatorRewards::new(&system_state.validators.validators);

        // Need to advance epoch so validator's staking starts counting
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Each validator gets 25 SOMA (total 100 SOMA rewards)
        advance_epoch_with_reward_amounts(&mut system_state, 100, &mut validator_stakes);

        // Validator total stake should increase by their share of rewards
        // Voting power: v1=1500, v2=2500, v3=3000, v4=3000
        // Rewards (100 SOMA): v1=15, v2=25, v3=30, v4=30
        assert_validator_total_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                115_000_000_000,
                225_000_000_000,
                330_000_000_000,
                430_000_000_000,
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

        // Voting power recalculated after v2's large stake increase:
        // v1=1304, v2=3174, v3=2486, v4=3036
        // Rewards (100 SOMA): v1=13.04, v2=31.74, v3=24.86, v4=30.36
        assert_validator_total_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                128_040_000_000,
                976_740_000_000,
                354_860_000_000,
                460_360_000_000,
            ],
        );
    }

    #[test]
    fn test_stake_subsidy() {
        let mut system_state = set_up_system_state_with_big_amounts();

        // Record initial validator states
        let mut validator_stakes = ValidatorRewards::new(&system_state.validators.validators);

        // Need to advance epoch so validator's staking starts counting
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Each validator gets 25 SOMA (total 100 SOMA rewards)
        advance_epoch_with_reward_amounts(&mut system_state, 100, &mut validator_stakes);

        // Validator total stake should increase by their share of rewards
        // Voting power: v1=1500, v2=2500, v3=3000, v4=3000
        // Rewards (100 SOMA): v1=15, v2=25, v3=30, v4=30
        assert_validator_total_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                100_000_015 * SHANNONS_PER_SOMA,
                200_000_025 * SHANNONS_PER_SOMA,
                300_000_030 * SHANNONS_PER_SOMA,
                400_000_030 * SHANNONS_PER_SOMA,
            ],
        );
    }

    #[test]
    fn test_stake_rewards() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Record initial validator states
        let mut validator_stakes = ValidatorRewards::new(&system_state.validators.validators);

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
        // Self-stake grows proportionally to their share of the pool
        // Voting power: v1=2452, v2=2452, v3=2451, v4=2645
        // Pool rewards: v1=29.424, v2=29.424, v3=29.412, v4=31.740 SOMA
        // v1 self: 100/300 * 29.424 = 9.808 → 109.808
        // v2 self: 200/300 * 29.424 = 19.616 → 219.616
        // v3 self: 300/300 * 29.412 = 29.412 → 329.412
        // v4 self: 400/400 * 31.740 = 31.740 → 431.740
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                109_808_000_000,
                219_616_000_000,
                329_412_000_000,
                431_740_000_000,
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
        // Withdrawal: 200 SOMA principal + 200/300 * 29.424 SOMA reward = 219.616 SOMA
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()),
            219_616_000_000
        );

        // Validator self-stake amounts after second round of 120 SOMA rewards
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                139_268_000_000,
                239_256_000_000,
                358_860_000_000,
                463_372_000_000,
            ],
            &validator_stakes,
        );

        // Unstake staker 2's first stake and track withdrawal
        let withdrawn_2 = unstake(&mut system_state, staked_soma_2);
        staker_withdrawals.insert(staker_addr_2(), withdrawn_2);

        // Verify staker 2's first withdrawal includes rewards
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_2()),
            119_628_000_000
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
        assert_eq!(staker_2_balance, 728_841_438_157);
    }

    #[test]
    fn test_stake_tiny_rewards() {
        let mut system_state = set_up_system_state_with_big_amounts();

        // Record initial validator states
        let mut validator_stakes = ValidatorRewards::new(&system_state.validators.validators);

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
        let mut validator_stakes = ValidatorRewards::new(&system_state.validators.validators);

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
        // v1 (0% commission): non-self = 100/200 * 22.488 = 11.244 → 111.244
        // v2 (20% commission): staker_reward = 32.508 * 0.8 = 26.0064, non-self = 100/300 * 26.0064 = 8.6688 → 108.6688
        assert_validator_non_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![111_244_000_000, 108_668_800_000, 0, 0],
            &validator_stakes,
        );

        // Check validator self stake amounts
        // v1: self 100/200 * 22.488 = 11.244 → 111.244
        // v2: self 200/300 * 26.0064 + 6.5016 commission = 17.3376 + 6.5016 = 23.8392 → 223.8392
        // v3: 300 + 32.496 = 332.496
        // v4: 400 + 32.508 = 432.508
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                111_244_000_000,
                223_839_200_000,
                332_496_000_000,
                432_508_000_000,
            ],
            &validator_stakes,
        );

        // Set commission rate for validator 1 to 10% (1000 basis points)
        set_commission_rate(&mut system_state, validator_addr_1(), 1000);

        // Advance epoch to apply commission rate change
        advance_epoch_with_reward_amounts(&mut system_state, 0, &mut validator_stakes);

        // Distribute more rewards (240 SOMA)
        advance_epoch_with_reward_amounts(&mut system_state, 240, &mut validator_stakes);

        // Verify total stake amounts after 240 SOMA more rewards
        assert_validator_total_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                267_800_000_000,
                397_404_000_000,
                397_392_000_000,
                497_404_000_000,
            ],
        );

        // Verify split between validator and staker stakes
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                136_165_600_000,
                271_767_980_095,
                397_392_000_000,
                497_404_000_000,
            ],
            &validator_stakes,
        );

        assert_validator_non_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![131_634_400_000, 125_636_019_905, 0, 0],
            &validator_stakes,
        );
    }

    #[test]
    fn test_rewards_slashing() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Record initial validator states
        let mut validator_stakes = ValidatorRewards::new(&system_state.validators.validators);

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

        // Validator 2 should have 10% of rewards slashed (reward_slashing_rate=1000 bps)
        // v2 reported by quorum → 10% of v2's reward redistributed to others
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                449_853_258_537,
                785_144_000_000,
                1_311_102_053_490,
                1_411_475_429_433,
            ],
            &validator_stakes,
        );

        // Unstake to check rewards
        let withdrawn_1 = unstake(&mut system_state, staked_soma_1);
        let withdrawn_2 = unstake(&mut system_state, staked_soma_2);

        staker_withdrawals.insert(staker_addr_1(), withdrawn_1);
        staker_withdrawals.insert(staker_addr_2(), withdrawn_2);

        // Staker 1 gets full rewards (same share as v1 self-stake)
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()),
            449_853_258_537
        );

        // Staker 2 gets reduced rewards (v2 was slashed 10%)
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_2()),
            392_572_000_000
        );
    }

    #[test]
    fn test_entire_rewards_slashing() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Record initial validator states
        let mut validator_stakes = ValidatorRewards::new(&system_state.validators.validators);

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

        // Validator 2 should have all rewards slashed (reward_slashing_rate=10000 bps = 100%)
        // v2's entire reward redistributed to other validators
        assert_validator_self_stake_amounts(
            &system_state,
            validator_addrs(),
            vec![
                562_652_585_379,
                200_000_000_000,
                1_637_100_534_906,
                1_737_594_294_335,
            ],
            &validator_stakes,
        );

        // Unstake to check rewards
        let withdrawn_1 = unstake(&mut system_state, staked_soma_1);
        let withdrawn_2 = unstake(&mut system_state, staked_soma_2);

        staker_withdrawals.insert(staker_addr_1(), withdrawn_1);
        staker_withdrawals.insert(staker_addr_2(), withdrawn_2);

        // Staker 1 gets enhanced rewards (redistribution from slashed v2)
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()),
            562_652_585_379
        );

        // Staker 2 only gets principal back (v2 was 100% slashed on rewards)
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_2()),
            100_000_000_000
        );
    }

    #[test]
    fn test_mul_rewards_withdraws_at_same_epoch() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Record initial validator states
        let mut validator_stakes = ValidatorRewards::new(&system_state.validators.validators);

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
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == validator_addr_1())
            .expect("Validator 1 not found");

        // Verify total stake in validator 1's pool
        // Rewards are proportional to voting power, not equal splits
        assert_eq!(
            validator_1.staking_pool.soma_balance,
            3_264_872_000_000,
            "Unexpected validator 1 pool balance"
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
        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_1()) / SHANNONS_PER_SOMA,
            435,
            "Incorrect withdrawal amount for staker 1"
        );

        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_2()) / SHANNONS_PER_SOMA,
            580,
            "Incorrect withdrawal amount for staker 2"
        );

        assert_eq!(
            total_soma_balance(&staker_withdrawals, staker_addr_3()) / SHANNONS_PER_SOMA,
            708,
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

        // Verify validator pool after all withdrawals
        let validator_1 = system_state
            .validators
            .validators
            .iter()
            .find(|v| v.metadata.soma_address == validator_addr_1())
            .expect("Validator 1 not found");

        assert_eq!(
            validator_1.staking_pool.soma_balance,
            140_929_659_227,
            "Unexpected validator 1 pool after all withdrawals"
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
        let mut system_state = test_utils::create_test_system_state(validators, 0, 0);

        // Record initial validator states
        let mut validator_stakes = ValidatorRewards::new(&system_state.validators.validators);

        // Double the stake of each validator through rewards
        advance_epoch_with_reward_amounts(&mut system_state, 10000, &mut validator_stakes);

        // Verify each validator's stake doubled
        for i in 0..num_validators {
            let addr = dbg_addr(i as u8 + 1);

            let validator = system_state
                .validators
                .validators
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
