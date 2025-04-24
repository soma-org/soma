#[cfg(test)]
mod encoder_staking_tests {
    use crate::{
        base::{dbg_addr, SomaAddress},
        config::genesis_config::SHANNONS_PER_SOMA,
        effects::ExecutionFailureStatus,
        error::SomaError,
        system_state::{
            encoder::Encoder,
            staking::StakedSoma,
            test_utils::{
                self, add_encoder, advance_epoch_with_reward_amounts, advance_epoch_with_rewards,
                assert_encoder_total_stake_amounts, create_encoder_for_testing,
                create_test_system_state, create_validator_for_testing, encoder_stake_amount,
                stake_with, stake_with_encoder, total_soma_balance, unstake,
                validator_stake_amount,
            },
            SystemParameters, SystemState,
        },
        transaction::UpdateEncoderMetadataArgs,
    };
    use std::collections::BTreeMap;

    // Create constant encoder addresses for testing
    fn encoder_addr_1() -> SomaAddress {
        dbg_addr(10)
    }
    fn encoder_addr_2() -> SomaAddress {
        dbg_addr(11)
    }
    fn new_encoder_addr() -> SomaAddress {
        dbg_addr(12)
    }

    // Create constant staker addresses for testing
    fn staker_addr_1() -> SomaAddress {
        dbg_addr(13)
    }
    fn staker_addr_2() -> SomaAddress {
        dbg_addr(14)
    }
    fn validator_addr_1() -> SomaAddress {
        dbg_addr(1)
    }

    // Helper to set up a test system state with encoders
    fn set_up_system_state() -> SystemState {
        let validators = vec![create_validator_for_testing(
            validator_addr_1(),
            100 * SHANNONS_PER_SOMA,
        )];

        let encoders = vec![
            create_encoder_for_testing(encoder_addr_1(), 100 * SHANNONS_PER_SOMA),
            create_encoder_for_testing(encoder_addr_2(), 100 * SHANNONS_PER_SOMA),
        ];

        create_test_system_state(validators, encoders, 1000, 0, 10, 500)
    }

    // Helper to set up a test system state with subsidy
    fn set_up_system_state_with_subsidy() -> SystemState {
        let validators = vec![create_validator_for_testing(
            validator_addr_1(),
            100 * SHANNONS_PER_SOMA,
        )];

        let encoders = vec![
            create_encoder_for_testing(encoder_addr_1(), 100 * SHANNONS_PER_SOMA),
            create_encoder_for_testing(encoder_addr_2(), 100 * SHANNONS_PER_SOMA),
        ];

        create_test_system_state(validators, encoders, 400, 0, 10, 0)
    }

    #[test]
    fn test_encoder_add_remove_stake_flow() {
        let mut system_state = set_up_system_state();

        // Create a stake to ENCODER_ADDR_1
        let staked_soma =
            stake_with_encoder(&mut system_state, staker_addr_1(), encoder_addr_1(), 60);

        // Verify initial stake amounts
        assert_eq!(
            encoder_stake_amount(&system_state, encoder_addr_1()).unwrap(),
            100 * SHANNONS_PER_SOMA
        );
        assert_eq!(
            encoder_stake_amount(&system_state, encoder_addr_2()).unwrap(),
            100 * SHANNONS_PER_SOMA
        );

        // Advance epoch to activate stake
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Verify stake has been added
        assert_eq!(
            encoder_stake_amount(&system_state, encoder_addr_1()).unwrap(),
            160 * SHANNONS_PER_SOMA
        );
        assert_eq!(
            encoder_stake_amount(&system_state, encoder_addr_2()).unwrap(),
            100 * SHANNONS_PER_SOMA
        );

        // Unstake from ENCODER_ADDR_1
        let unstaked_amount = unstake(&mut system_state, staked_soma);
        assert_eq!(unstaked_amount, 60 * SHANNONS_PER_SOMA);

        // Verify stake is still present until next epoch
        assert_eq!(
            encoder_stake_amount(&system_state, encoder_addr_1()).unwrap(),
            160 * SHANNONS_PER_SOMA
        );

        // Advance epoch to process withdrawal
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Verify stake has been removed
        assert_eq!(
            encoder_stake_amount(&system_state, encoder_addr_1()).unwrap(),
            100 * SHANNONS_PER_SOMA
        );
    }

    #[test]
    fn test_encoder_reporting_mechanism() {
        let mut system_state = set_up_system_state();

        // Test encoder reporting encoder
        let report_result = system_state.report_encoder(encoder_addr_2(), encoder_addr_1());
        assert!(
            report_result.is_ok(),
            "Encoder should be able to report another encoder"
        );

        // Verify reporters are recorded
        let reporters = system_state
            .encoder_report_records
            .get(&encoder_addr_1())
            .unwrap();

        assert!(
            reporters.contains(&encoder_addr_2()),
            "Encoder should be in reporter set"
        );

        // Test undo report
        let undo_result = system_state.undo_report_encoder(encoder_addr_2(), encoder_addr_1());
        assert!(undo_result.is_ok(), "Should be able to undo report");

        // Verify record updated
        let reporters = system_state.encoder_report_records.get(&encoder_addr_1());

        assert!(reporters.is_none(), "Encoder should not be in reporter set");
    }

    #[test]
    fn test_validator_receives_all_rewards() {
        let mut system_state = set_up_system_state_with_subsidy();

        // Initial validator and encoder stake
        let initial_validator_stake =
            validator_stake_amount(&system_state, validator_addr_1()).unwrap();
        let initial_encoder_stake_1 =
            encoder_stake_amount(&system_state, encoder_addr_1()).unwrap();
        let initial_encoder_stake_2 =
            encoder_stake_amount(&system_state, encoder_addr_2()).unwrap();

        // Advance epoch with transaction fees (200 SOMA)
        let _ = advance_epoch_with_rewards(&mut system_state, 200 * SHANNONS_PER_SOMA).unwrap();

        // Check validator and encoder stakes after rewards
        let new_validator_stake =
            validator_stake_amount(&system_state, validator_addr_1()).unwrap();
        let new_encoder_stake_1 = encoder_stake_amount(&system_state, encoder_addr_1()).unwrap();
        let new_encoder_stake_2 = encoder_stake_amount(&system_state, encoder_addr_2()).unwrap();

        // Validator should receive all rewards
        assert!(
            new_validator_stake > initial_validator_stake,
            "Validator stake should increase from rewards"
        );

        // Encoders should not receive any rewards
        assert_eq!(
            new_encoder_stake_1, initial_encoder_stake_1,
            "Encoder 1 should not receive rewards"
        );
        assert_eq!(
            new_encoder_stake_2, initial_encoder_stake_2,
            "Encoder 2 should not receive rewards"
        );

        // Calculate expected validator reward amount
        let validator_reward = new_validator_stake - initial_validator_stake;
        assert_eq!(
            validator_reward,
            200 * SHANNONS_PER_SOMA,
            "Validator should receive full reward amount"
        );
    }

    #[test]
    fn test_add_encoder_candidate() {
        let mut system_state = set_up_system_state();

        // Add a new encoder
        add_encoder(&mut system_state, new_encoder_addr());

        // Verify encoder is in pending_active_encoders
        assert!(system_state.encoders.is_pending_encoder(new_encoder_addr()));
        assert!(!system_state.encoders.is_active_encoder(new_encoder_addr()));

        // Add stake to pending encoder
        let staked_soma =
            stake_with_encoder(&mut system_state, staker_addr_1(), new_encoder_addr(), 200);

        // Advance epoch to activate encoder
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Verify encoder is now active with stake
        assert!(system_state.encoders.is_active_encoder(new_encoder_addr()));
        assert_eq!(
            encoder_stake_amount(&system_state, new_encoder_addr()).unwrap(),
            200 * SHANNONS_PER_SOMA
        );
    }

    #[test]
    fn test_remove_encoder() {
        let mut system_state = set_up_system_state();

        // Add stake to encoder
        let staked_soma =
            stake_with_encoder(&mut system_state, staker_addr_1(), encoder_addr_1(), 100);

        // Advance epoch to activate stake
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Request to remove the encoder
        system_state
            .request_remove_encoder(encoder_addr_1())
            .expect("Failed to request encoder removal");

        // Encoder should still be active until next epoch
        assert!(system_state.encoders.is_active_encoder(encoder_addr_1()));

        // Advance epoch to process removal
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Encoder should no longer be active
        assert!(!system_state.encoders.is_active_encoder(encoder_addr_1()));

        // Unstake should still work from inactive encoder
        let withdrawn = unstake(&mut system_state, staked_soma);
        assert_eq!(withdrawn, 100 * SHANNONS_PER_SOMA);
    }
    #[test]
    fn test_encoder_metadata_update() {
        let mut system_state = set_up_system_state();

        // Create metadata update args - ensure we properly BCS serialize the string
        let network_address = "/ip4/127.0.0.1/tcp/9000".to_string();
        let encoded_address = bcs::to_bytes(&network_address).expect("Failed to BCS serialize");

        let args = UpdateEncoderMetadataArgs {
            next_epoch_network_address: Some(encoded_address),
            next_epoch_network_pubkey: None,
        };

        // Request metadata update
        system_state
            .request_update_encoder_metadata(encoder_addr_1(), &args)
            .expect("Failed to update encoder metadata");

        // Encoder metadata should not change until next epoch
        let encoder_before = system_state
            .encoders
            .find_encoder(encoder_addr_1())
            .unwrap();
        let old_address = encoder_before.metadata.net_address.clone();

        // Advance epoch to apply changes
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Check that metadata was updated
        let encoder_after = system_state
            .encoders
            .find_encoder(encoder_addr_1())
            .unwrap();
        assert_ne!(encoder_after.metadata.net_address, old_address);
    }

    #[test]
    fn test_stake_delegation_across_types() {
        let mut system_state = set_up_system_state();
        let mut staker_withdrawals: BTreeMap<SomaAddress, u64> = BTreeMap::new();

        // Stake to both encoder and validator
        let encoder_staked_soma =
            stake_with_encoder(&mut system_state, staker_addr_1(), encoder_addr_1(), 50);
        let validator_staked_soma =
            stake_with(&mut system_state, staker_addr_1(), validator_addr_1(), 50);

        // Advance epoch to activate stakes
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Verify stakes were properly added
        assert_eq!(
            encoder_stake_amount(&system_state, encoder_addr_1()).unwrap(),
            150 * SHANNONS_PER_SOMA // 100 initial + 50 staked
        );
        assert_eq!(
            validator_stake_amount(&system_state, validator_addr_1()).unwrap(),
            150 * SHANNONS_PER_SOMA // 100 initial + 50 staked
        );

        // Distribute rewards (should all go to validator)
        let _ = advance_epoch_with_rewards(&mut system_state, 100 * SHANNONS_PER_SOMA).unwrap();

        // Unstake from both
        let withdrawn_encoder = unstake(&mut system_state, encoder_staked_soma);
        let withdrawn_validator = unstake(&mut system_state, validator_staked_soma);

        // Staker should get principal plus rewards from validator only
        staker_withdrawals.insert(staker_addr_1(), withdrawn_encoder + withdrawn_validator);

        // Encoder stake return should just be principal
        assert_eq!(withdrawn_encoder, 50 * SHANNONS_PER_SOMA);

        // Validator stake return should include rewards
        assert!(withdrawn_validator > 50 * SHANNONS_PER_SOMA);
    }
}
