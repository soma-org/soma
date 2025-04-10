#[cfg(test)]
mod tests {
    use crate::{
        base::SomaAddress,
        multiaddr::Multiaddr,
        system_state::{
            staking::{PoolTokenExchangeRate, StakedSoma, StakingPool},
            subsidy::StakeSubsidy,
            test_utils::*,
            validator::{Validator, ValidatorSet},
            PublicKey, SystemParameters, SystemState,
        },
    };
    use std::str::FromStr;

    #[test]
    fn test_distribute_rewards_and_advance_epoch() {
        // Create a system state with one validator with 1000 SOMA initial stake
        let validator_address = SomaAddress::random();
        let validator = create_validator_for_testing(validator_address, 1000 * SHANNONS_PER_SOMA);
        let validators = vec![validator];

        // Create system state with subsidy of 100 SOMA per epoch
        let mut system_state = create_test_system_state(
            validators, 10000, // 10,000 SOMA subsidy fund
            100,   // 100 SOMA initial distribution
            10,    // period length of 10 epochs
            500,   // 5% decrease rate
        );

        // Verify initial state
        assert_eq!(system_state.epoch, 0);
        let initial_validator = &system_state.validators.active_validators[0];
        let initial_stake = initial_validator.staking_pool.soma_balance;
        let initial_rewards_pool = initial_validator.staking_pool.rewards_pool;

        assert_eq!(initial_stake, 1000 * SHANNONS_PER_SOMA);
        assert_eq!(initial_rewards_pool, 0);
        assert_eq!(
            system_state.stake_subsidy.current_distribution_amount,
            100 * SHANNONS_PER_SOMA
        );

        // Advance to epoch 1 and distribute rewards
        let new_epoch = distribute_rewards_and_advance_epoch(&mut system_state, 0);
        assert_eq!(new_epoch, 1);

        // Check that rewards were distributed
        let validator = &system_state.validators.active_validators[0];
        let total_stake_after_rewards = validator.staking_pool.soma_balance;

        // Total stake should have increased by 100 SOMA (subsidy amount)
        assert_eq!(
            total_stake_after_rewards - initial_stake,
            100 * SHANNONS_PER_SOMA,
            "Expected stake to increase by 100 SOMA"
        );

        // Check that rewards were split correctly based on commission rate (10%)
        // Commission rate is 1000 basis points (10%)
        let total_reward = total_stake_after_rewards - initial_stake;
        let expected_validator_commission = total_reward * 10 / 100;
        let expected_staker_reward = total_reward - expected_validator_commission;

        // Rewards pool should have staker portion
        assert_eq!(
            validator.staking_pool.rewards_pool, expected_staker_reward,
            "Rewards pool should contain staker portion"
        );

        // Check subsidy counter was incremented
        assert_eq!(system_state.stake_subsidy.distribution_counter, 1);

        // Advance through multiple epochs to check subsidy decay
        for i in 2..12 {
            distribute_rewards_and_advance_epoch(&mut system_state, 0);
        }

        // After 10 more epochs (total 11), subsidy should have decreased once
        assert_eq!(system_state.stake_subsidy.distribution_counter, 11);

        // Subsidy should have decreased by 5%
        let original_amount = 100 * SHANNONS_PER_SOMA;
        let expected_decreased_amount = original_amount - (original_amount * 5 / 100);

        assert_eq!(
            system_state.stake_subsidy.current_distribution_amount, expected_decreased_amount,
            "Subsidy should decrease by 5% after period length"
        );
    }

    #[test]
    fn test_multiple_validators_reward_distribution() {
        // Create system state with three validators of different sizes
        let validator1 =
            create_validator_for_testing(SomaAddress::random(), 1000 * SHANNONS_PER_SOMA); // 1000 SOMA
        let validator2 =
            create_validator_for_testing(SomaAddress::random(), 2000 * SHANNONS_PER_SOMA); // 2000 SOMA
        let validator3 =
            create_validator_for_testing(SomaAddress::random(), 3000 * SHANNONS_PER_SOMA); // 3000 SOMA

        let validators = vec![validator1, validator2, validator3];
        let validator_addresses: Vec<SomaAddress> =
            validators.iter().map(|v| v.metadata.soma_address).collect();

        // Total stake is 6000 SOMA
        let total_initial_stake = 6000 * SHANNONS_PER_SOMA;

        // Create system state with 300 SOMA per epoch subsidy
        let mut system_state = create_test_system_state(
            validators, 10000, // 10,000 SOMA subsidy fund
            300,   // 300 SOMA initial distribution
            10,    // period length of 10 epochs
            500,   // 5% decrease rate
        );

        // Verify initial state
        assert_eq!(system_state.epoch, 0);
        assert_eq!(system_state.validators.total_stake, total_initial_stake);

        // Each validator should have voting power proportional to their stake
        assert_eq!(
            system_state.validators.active_validators[0].voting_power,
            (1000
                * SHANNONS_PER_SOMA as u128
                * system_state.validators.active_validators[0].voting_power as u128
                / total_initial_stake as u128) as u64
        );

        assert_eq!(
            system_state.validators.active_validators[1].voting_power,
            (2000
                * SHANNONS_PER_SOMA as u128
                * system_state.validators.active_validators[1].voting_power as u128
                / total_initial_stake as u128) as u64
        );

        assert_eq!(
            system_state.validators.active_validators[2].voting_power,
            (3000
                * SHANNONS_PER_SOMA as u128
                * system_state.validators.active_validators[2].voting_power as u128
                / total_initial_stake as u128) as u64
        );

        // Advance to epoch 1 and distribute rewards
        distribute_rewards_and_advance_epoch(&mut system_state, 0);

        // Extract balances after rewards
        let stake1 = system_state.validators.active_validators[0]
            .staking_pool
            .soma_balance;
        let stake2 = system_state.validators.active_validators[1]
            .staking_pool
            .soma_balance;
        let stake3 = system_state.validators.active_validators[2]
            .staking_pool
            .soma_balance;

        // Total rewards should be 300 SOMA
        let total_rewards = (stake1 + stake2 + stake3) - total_initial_stake;
        assert_eq!(total_rewards, 300 * SHANNONS_PER_SOMA);

        // Each validator should get rewards proportional to stake
        // (with small rounding errors allowed)
        let expected_reward1 = 300 * SHANNONS_PER_SOMA * 1000 / 6000;
        let expected_reward2 = 300 * SHANNONS_PER_SOMA * 2000 / 6000;
        let expected_reward3 = 300 * SHANNONS_PER_SOMA * 3000 / 6000;

        assert!(
            (stake1 - 1000 * SHANNONS_PER_SOMA).abs_diff(expected_reward1) <= 1,
            "Validator 1 reward incorrect"
        );

        assert!(
            (stake2 - 2000 * SHANNONS_PER_SOMA).abs_diff(expected_reward2) <= 1,
            "Validator 2 reward incorrect"
        );

        assert!(
            (stake3 - 3000 * SHANNONS_PER_SOMA).abs_diff(expected_reward3) <= 1,
            "Validator 3 reward incorrect"
        );
    }

    #[test]
    fn test_exchange_rate_update() {
        // Create a validator with initial stake
        let validator =
            create_validator_for_testing(SomaAddress::random(), 1000 * SHANNONS_PER_SOMA);
        let validators = vec![validator];

        // Create system state with subsidy
        let mut system_state = create_test_system_state(
            validators, 10000, // 10,000 SOMA subsidy fund
            100,   // 100 SOMA initial distribution
            10,    // period length of 10 epochs
            500,   // 5% decrease rate
        );

        // Activate validator's staking pool
        system_state.validators.active_validators[0].activate(0);

        // Initial exchange rate should be 1:1
        let validator = &system_state.validators.active_validators[0];
        let initial_exchange_rate = validator.staking_pool.pool_token_exchange_rate_at_epoch(0);

        assert_eq!(initial_exchange_rate.soma_amount, 1000 * SHANNONS_PER_SOMA);
        assert_eq!(
            initial_exchange_rate.pool_token_amount,
            1000 * SHANNONS_PER_SOMA
        );

        // Advance epoch and distribute rewards
        distribute_rewards_and_advance_epoch(&mut system_state, 0);

        // Exchange rate should now reflect rewards (1100 SOMA for 1000 pool tokens)
        let validator = &system_state.validators.active_validators[0];
        let exchange_rate_epoch_1 = validator.staking_pool.pool_token_exchange_rate_at_epoch(1);

        assert_eq!(exchange_rate_epoch_1.soma_amount, 1100 * SHANNONS_PER_SOMA);
        assert_eq!(
            exchange_rate_epoch_1.pool_token_amount,
            1000 * SHANNONS_PER_SOMA
        );

        // Add stake during epoch 1
        let mut staked_soma = system_state
            .request_add_stake(
                SomaAddress::random(),           // Staker address
                validator.metadata.soma_address, // Validator address
                900 * SHANNONS_PER_SOMA,         // Stake amount
            )
            .unwrap();

        // Advance to epoch 2
        distribute_rewards_and_advance_epoch(&mut system_state, 0);

        // The new stake should be active and pool tokens issued at the epoch 1 rate
        // 900 SOMA at 1.1 SOMA per token = ~818.18 pool tokens
        let validator = &system_state.validators.active_validators[0];
        let exchange_rate_epoch_2 = validator.staking_pool.pool_token_exchange_rate_at_epoch(2);

        // With rewards from epoch 2 (100 SOMA), total is 2100 SOMA
        // Pool tokens should be 1000 (original) + 818.18 (new) ≈ 1818.18

        // Allow small rounding differences
        assert!(
            (exchange_rate_epoch_2.soma_amount - 2100 * SHANNONS_PER_SOMA).abs_diff(0) <= 1,
            "Total SOMA amount incorrect"
        );

        // Expected pool tokens calculated based on the exchange rate at epoch 1
        let expected_new_pool_tokens = ((((900 * SHANNONS_PER_SOMA) as f64)
            * ((1000 * SHANNONS_PER_SOMA) as f64))
            / ((1100 * SHANNONS_PER_SOMA) as f64)) as u64;

        assert!(
            (exchange_rate_epoch_2.pool_token_amount
                - (1000 * SHANNONS_PER_SOMA + expected_new_pool_tokens))
                .abs_diff(0)
                <= 2,
            "Pool token amount incorrect"
        );

        // Check exchange rate calculation
        let soma_per_token = exchange_rate_epoch_2.soma_amount as f64
            / exchange_rate_epoch_2.pool_token_amount as f64;
        assert!(
            (soma_per_token - 1.15).abs() < 0.01,
            "Exchange rate should be approximately 1.15 SOMA per token"
        );
    }

    #[test]
    fn test_staking_and_rewards() {
        // Create a validator with initial stake
        let validator =
            create_validator_for_testing(SomaAddress::random(), 1000 * SHANNONS_PER_SOMA);
        let validator_address = validator.metadata.soma_address;
        let validators = vec![validator];

        // Create system state with subsidy
        let mut system_state = create_test_system_state(
            validators, 10000, // 10,000 SOMA subsidy fund
            100,   // 100 SOMA initial distribution
            10,    // period length of 10 epochs
            500,   // 5% decrease rate
        );

        // Activate validator's staking pool
        system_state.validators.active_validators[0].activate(0);

        // Advance epoch and distribute rewards
        distribute_rewards_and_advance_epoch(&mut system_state, 0);

        // Add stake from a delegator
        let staker_address = SomaAddress::random();
        let mut staked_soma = stake_with(
            &mut system_state,
            staker_address,
            validator_address,
            500, // 500 SOMA
        )
        .unwrap();

        // Stake should be pending until next epoch
        assert_eq!(
            system_state.validators.active_validators[0]
                .staking_pool
                .pending_stake,
            500 * SHANNONS_PER_SOMA
        );

        // Advance to next epoch to activate stake
        distribute_rewards_and_advance_epoch(&mut system_state, 0);

        // Pending stake should be processed
        assert_eq!(
            system_state.validators.active_validators[0]
                .staking_pool
                .pending_stake,
            0
        );

        // Advance a few more epochs to accumulate rewards
        for _ in 0..3 {
            distribute_rewards_and_advance_epoch(&mut system_state, 0);
        }

        // Withdraw stake and check rewards
        let withdrawn_amount = unstake(&mut system_state, staked_soma).unwrap();

        // Withdrawn amount should be greater than principal due to rewards
        assert!(
            withdrawn_amount > 500 * SHANNONS_PER_SOMA,
            "Withdrawn amount should include rewards"
        );

        // Expected rewards based on exchange rate changes
        // Initial exchange rate at epoch 1: 1.1 SOMA per token
        // 500 SOMA would get approximately 454.55 pool tokens
        // After 3 more epochs with 100 SOMA each, exchange rate should be higher
        // Exchange rate = (1100 + 500 + 3*100) / (1000 + 454.55) ≈ 1.3 SOMA per token
        // So 454.55 tokens should be worth approximately 590.91 SOMA

        // Allow for some rounding error
        assert!(
            withdrawn_amount > 590 * SHANNONS_PER_SOMA
                && withdrawn_amount < 600 * SHANNONS_PER_SOMA,
            "Withdrawn amount should be approximately 590 SOMA but was {}",
            withdrawn_amount / SHANNONS_PER_SOMA
        );
    }

    #[test]
    fn test_commission_rate_change() {
        // Create a validator with initial stake and 10% commission
        let mut validator =
            create_validator_for_testing(SomaAddress::random(), 1000 * SHANNONS_PER_SOMA);
        validator.commission_rate = 1000; // 10% in basis points
        let validator_address = validator.metadata.soma_address;
        let validators = vec![validator];

        // Create system state with subsidy
        let mut system_state = create_test_system_state(
            validators, 10000, // 10,000 SOMA subsidy fund
            100,   // 100 SOMA initial distribution
            10,    // period length of 10 epochs
            500,   // 5% decrease rate
        );

        // Advance epoch with 10% commission
        distribute_rewards_and_advance_epoch(&mut system_state, 0);

        // Check reward split with 10% commission
        let validator = &system_state.validators.active_validators[0];
        let total_reward = validator.staking_pool.soma_balance - 1000 * SHANNONS_PER_SOMA;
        let staker_reward = validator.staking_pool.rewards_pool;
        let validator_commission = total_reward - staker_reward;

        assert_eq!(
            validator_commission,
            total_reward * 10 / 100,
            "Validator commission should be 10% of rewards"
        );

        // Request commission rate change to 20%
        system_state
            .request_set_commission_rate(validator_address, 2000)
            .unwrap();

        // Commission rate should be staged for next epoch
        assert_eq!(
            system_state.validators.active_validators[0].next_epoch_commission_rate,
            2000
        );

        // Advance epoch to apply new commission rate
        distribute_rewards_and_advance_epoch(&mut system_state, 0);

        // Check that commission rate was updated
        assert_eq!(
            system_state.validators.active_validators[0].commission_rate, 2000,
            "Commission rate should be updated to 20%"
        );

        // Record reward pool before next epoch
        let reward_pool_before = system_state.validators.active_validators[0]
            .staking_pool
            .rewards_pool;

        // Advance epoch with new 20% commission
        distribute_rewards_and_advance_epoch(&mut system_state, 0);

        // Check reward split with 20% commission
        let validator = &system_state.validators.active_validators[0];
        let new_total_reward =
            validator.staking_pool.soma_balance - (1000 * SHANNONS_PER_SOMA + 2 * total_reward);

        let new_staker_reward = validator.staking_pool.rewards_pool - reward_pool_before;
        let new_validator_commission = new_total_reward - new_staker_reward;

        // Allow small rounding differences
        let expected_commission = new_total_reward * 20 / 100;
        assert!(
            (new_validator_commission as i64 - expected_commission as i64).abs() <= 1,
            "Validator commission should be 20% of rewards"
        );
    }
}
