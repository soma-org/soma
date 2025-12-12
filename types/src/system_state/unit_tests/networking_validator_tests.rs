#[cfg(test)]
mod networking_validator_tests {
    use crate::{
        base::{dbg_addr, SomaAddress},
        committee::{
            TOTAL_VOTING_POWER, VALIDATOR_CONSENSUS_LOW_POWER, VALIDATOR_CONSENSUS_MIN_POWER,
            VALIDATOR_CONSENSUS_VERY_LOW_POWER, VALIDATOR_LOW_STAKE_GRACE_PERIOD,
            VALIDATOR_NETWORKING_MIN_POWER,
        },
        config::genesis_config::SHANNONS_PER_SOMA,
        effects::ExecutionFailureStatus,
        system_state::{
            staking::StakedSoma,
            test_utils::{
                add_validator, advance_epoch_with_rewards, create_test_system_state,
                create_validator_for_testing, stake_with, unstake, validator_stake_amount,
                ValidatorRewards,
            },
            validator::Validator,
            SystemState, SystemStateTrait,
        },
    };
    use std::collections::{BTreeMap, HashMap};

    // Helper addresses
    fn validator_addr(i: u8) -> SomaAddress {
        dbg_addr(i)
    }

    fn staker_addr(i: u8) -> SomaAddress {
        dbg_addr(100 + i)
    }

    // Create system state with validators at different stake levels
    // Key thresholds:
    // - Voting power < 4: Immediate demotion to networking
    // - Voting power 4-7: At risk with grace period
    // - Voting power 8-11: Safe in current tier (hysteresis band)
    // - Voting power >= 12: Can join/stay in consensus
    fn create_tiered_system_state() -> SystemState {
        let validators = vec![
            // Consensus validator - massive stake to dilute others
            create_validator_for_testing(validator_addr(1), 50_000 * SHANNONS_PER_SOMA),
            // Consensus validator - will be at risk when diluted
            create_validator_for_testing(validator_addr(2), 400 * SHANNONS_PER_SOMA),
            // This validator should be IMMEDIATELY demoted (voting power < 4)
            // With ~100,000 total stake, need < 40 SOMA for voting power < 4
            create_validator_for_testing(validator_addr(3), 35 * SHANNONS_PER_SOMA),
            // Another consensus validator
            create_validator_for_testing(validator_addr(4), 50_000 * SHANNONS_PER_SOMA),
        ];

        create_test_system_state(validators, vec![], 1000, 0)
    }

    #[test]
    fn test_initial_validator_placement() {
        let mut system_state = create_tiered_system_state();

        // Initially all validators start as consensus (genesis behavior)
        assert_eq!(system_state.validators.consensus_validators.len(), 4);
        assert_eq!(system_state.validators.networking_validators.len(), 0);

        // Advance epoch to trigger transitions
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // After epoch transition, validator 3 should be demoted to networking
        // Total stake: 50,000 + 400 + 35 + 50,000 = 100,435 SOMA
        // Validator 3 voting power: (35/100,435) * 10,000 ≈ 3.48 (below 4 - immediate demotion)
        assert_eq!(
            system_state.validators.consensus_validators.len(),
            3,
            "Should have 3 consensus validators after demotion"
        );
        assert_eq!(
            system_state.validators.networking_validators.len(),
            1,
            "Should have 1 networking validator after demotion"
        );

        // Verify validator 3 is in networking
        assert!(
            system_state
                .validators
                .is_networking_validator(validator_addr(3)),
            "Validator 3 should be in networking tier"
        );
        assert!(
            !system_state
                .validators
                .is_consensus_validator(validator_addr(3)),
            "Validator 3 should not be in consensus tier"
        );
    }

    #[test]
    fn test_networking_to_consensus_promotion() {
        let mut system_state = create_tiered_system_state();

        // Advance to apply initial placement
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Validator 3 starts in networking (voting power ~3.48)
        assert!(
            system_state
                .validators
                .is_networking_validator(validator_addr(3)),
            "Validator 3 should start in networking"
        );

        // Add stake to validator 3 to increase voting power above MIN_POWER (12)
        // Total stake: ~100,435 SOMA
        // Need stake >= (12 * 100,435) / 10,000 ≈ 120.5 SOMA
        // Current stake: 35 SOMA, need to add at least 86 SOMA
        let _ = stake_with(&mut system_state, staker_addr(1), validator_addr(3), 100);

        // Advance epoch to activate stake
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Validator 3 should be promoted to consensus
        // New stake: 35 + 100 = 135 SOMA
        // Total stake: ~100,535 SOMA
        // Voting power: (135/100,535) * 10,000 ≈ 13.4 (above 12)
        assert!(
            system_state
                .validators
                .is_consensus_validator(validator_addr(3)),
            "Validator 3 should be promoted to consensus"
        );
        assert!(
            !system_state
                .validators
                .is_networking_validator(validator_addr(3)),
            "Validator 3 should no longer be in networking"
        );
    }

    #[test]
    fn test_grace_period_for_at_risk_validators() {
        let mut system_state = create_tiered_system_state();

        // Advance to apply initial placement
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Validator 2 has 400 SOMA out of ~100,435 total
        // Voting power: (400/100,435) * 10,000 ≈ 39.8 (well above all thresholds)
        // We need to dilute it into at-risk range (below 8, above 4)

        // Add stake to dilute validator 2 into at-risk range
        // Need total stake such that: 4 < (400/total) * 10,000 < 8
        // This means: 50,000 < total < 100,000
        // Current total: ~100,435, validator 2 is at ~39.8 voting power
        // Need to add stake to make total ~450,000 for voting power ~8.9
        let _ = stake_with(
            &mut system_state,
            staker_addr(1),
            validator_addr(1),
            350_000,
        );

        // Advance epoch
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Total stake now: ~450,435 SOMA
        // Validator 2 voting power: (400/450,435) * 10,000 ≈ 8.9 (still above 8)
        // Need more dilution
        let _ = stake_with(
            &mut system_state,
            staker_addr(2),
            validator_addr(4),
            100_000,
        );
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Total stake now: ~550,435 SOMA
        // Validator 2 voting power: (400/550,435) * 10,000 ≈ 7.3 (between 4 and 8)

        // Validator 2 should be at risk
        assert!(
            system_state
                .validators
                .at_risk_validators
                .contains_key(&validator_addr(2)),
            "Validator 2 should be at risk"
        );

        // Still in consensus during grace period
        assert!(
            system_state
                .validators
                .is_consensus_validator(validator_addr(2)),
            "Should still be consensus during grace period"
        );

        // Advance through grace period
        for i in 0..VALIDATOR_LOW_STAKE_GRACE_PERIOD {
            let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

            if i < VALIDATOR_LOW_STAKE_GRACE_PERIOD - 1 {
                assert!(
                    system_state
                        .validators
                        .is_consensus_validator(validator_addr(2)),
                    "Should still be consensus during grace period epoch {}",
                    i
                );
            }
        }

        // After grace period expires, should be demoted to networking
        assert!(
            !system_state
                .validators
                .is_consensus_validator(validator_addr(2)),
            "Should be demoted from consensus after grace period"
        );
        assert!(
            system_state
                .validators
                .is_networking_validator(validator_addr(2)),
            "Should be in networking after grace period"
        );
    }

    #[test]
    fn test_consensus_to_networking_demotion() {
        let mut system_state = create_tiered_system_state();

        // Advance to apply initial placement
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Validator 2 starts in consensus with 400 SOMA
        assert!(system_state
            .validators
            .is_consensus_validator(validator_addr(2)));

        // Add massive stake to dilute validator 2 below LOW_POWER (8)
        // Need total stake such that: (400/total) * 10,000 < 8
        // total > 400 * 10,000 / 8 = 500,000 SOMA
        let _ = stake_with(
            &mut system_state,
            staker_addr(1),
            validator_addr(1),
            450_000,
        );

        // Advance epoch to activate stake
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Total stake now: ~550,435 SOMA
        // Validator 2 voting power: (400/550,435) * 10,000 ≈ 7.3 (below 8, above 4)

        // Validator 2 should enter at-risk period
        assert!(
            system_state
                .validators
                .at_risk_validators
                .contains_key(&validator_addr(2)),
            "Validator 2 should be at risk"
        );

        // Advance through grace period
        for _ in 0..VALIDATOR_LOW_STAKE_GRACE_PERIOD {
            let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();
        }

        // After grace period, should be demoted
        assert!(
            !system_state
                .validators
                .is_consensus_validator(validator_addr(2)),
            "Validator 2 should be demoted from consensus"
        );
        assert!(
            system_state
                .validators
                .is_networking_validator(validator_addr(2)),
            "Validator 2 should be in networking"
        );
    }

    #[test]
    fn test_immediate_demotion_below_very_low_threshold() {
        let mut system_state = create_tiered_system_state();

        // Advance to apply initial placement
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Add massive stake to dilute validator 2 below VERY_LOW_POWER (4)
        // Need: (400/total) * 10,000 < 4
        // total > 400 * 10,000 / 4 = 1,000,000 SOMA
        let _ = stake_with(
            &mut system_state,
            staker_addr(1),
            validator_addr(1),
            950_000,
        );

        // Advance epoch
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Total stake: ~1,050,435 SOMA
        // Validator 2 voting power: (400/1,050,435) * 10,000 ≈ 3.8 (below 4)

        // Should be immediately demoted (no grace period)
        assert!(
            !system_state
                .validators
                .is_consensus_validator(validator_addr(2)),
            "Validator 2 should be immediately demoted"
        );
        assert!(
            system_state
                .validators
                .is_networking_validator(validator_addr(2)),
            "Validator 2 should be in networking"
        );
    }

    #[test]
    fn test_networking_to_inactive_removal() {
        // Use different initial setup for this test
        let validators = vec![
            create_validator_for_testing(validator_addr(1), 500_000 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr(2), 100_000 * SHANNONS_PER_SOMA),
            // This will start with voting power ~8.3 (networking after first epoch)
            create_validator_for_testing(validator_addr(3), 500 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr(4), 400_000 * SHANNONS_PER_SOMA),
        ];

        let mut system_state = create_test_system_state(validators, vec![], 1000, 0);

        // Advance to apply initial placement
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Check if validator 3 is in the right tier
        // Total: 1,000,500 SOMA, Validator 3: 500 SOMA
        // Voting power: (500/1,000,500) * 10,000 ≈ 5.0
        // This is between 4 and 8, so it will be at risk, not immediately networking

        // Force it to networking by advancing through grace period
        for _ in 0..=VALIDATOR_LOW_STAKE_GRACE_PERIOD {
            let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();
        }

        // Now it should be in networking
        assert!(
            system_state
                .validators
                .is_networking_validator(validator_addr(3)),
            "Validator 3 should be in networking"
        );

        // Add massive stake to dilute further below networking threshold
        let _ = stake_with(
            &mut system_state,
            staker_addr(1),
            validator_addr(1),
            10_000_000,
        );
        let _ = stake_with(
            &mut system_state,
            staker_addr(2),
            validator_addr(4),
            10_000_000,
        );

        // Advance epoch
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Total stake: ~21,000,500 SOMA
        // Validator 3 voting power: (500/21,000,500) * 10,000 ≈ 0.24 (below 1)
        assert!(!system_state
            .validators
            .is_consensus_validator(validator_addr(3)));
        assert!(!system_state
            .validators
            .is_networking_validator(validator_addr(3)));
        assert!(!system_state
            .validators
            .is_active_validator(validator_addr(3)));
    }
}
