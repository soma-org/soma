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
            SystemParameters, SystemState, SystemStateTrait,
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

        create_test_system_state(validators, encoders, 1000, 0)
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

        create_test_system_state(validators, encoders, 400, 0)
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
            next_epoch_internal_network_address: Some(encoded_address),
            next_epoch_external_network_address: None,
            next_epoch_network_pubkey: None,
            next_epoch_object_server_address: None,
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
        let old_address = encoder_before.metadata.internal_network_address.clone();

        // Advance epoch to apply changes
        let _ = advance_epoch_with_rewards(&mut system_state, 0).unwrap();

        // Check that metadata was updated
        let encoder_after = system_state
            .encoders
            .find_encoder(encoder_addr_1())
            .unwrap();
        assert_ne!(encoder_after.metadata.internal_network_address, old_address);
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

    #[test]
    fn test_encoder_reference_byte_price_derivation() {
        use crate::{
            base::dbg_addr,
            config::genesis_config::SHANNONS_PER_SOMA,
            system_state::{
                encoder::Encoder,
                test_utils::{
                    add_encoder, advance_epoch_with_rewards, create_encoder_for_testing,
                    create_test_system_state, create_validator_for_testing, encoder_stake_amount,
                    set_up_system_state, stake_with, stake_with_encoder, validator_stake_amount,
                },
            },
        };

        // Create constant addresses for testing
        let validator_addr_1 = dbg_addr(1);
        let validator_addr_2 = dbg_addr(2);
        let encoder_addr_1 = dbg_addr(10);
        let encoder_addr_2 = dbg_addr(11);
        let encoder_addr_3 = dbg_addr(12);
        let encoder_addr_4 = dbg_addr(13);
        let encoder_addr_5 = dbg_addr(14);
        let staker_addr = dbg_addr(20);

        // Create validators for the system
        let validators = vec![
            create_validator_for_testing(validator_addr_1, 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr_2, 150 * SHANNONS_PER_SOMA),
        ];

        // Create encoders with different stakes and byte prices
        let mut encoders = vec![
            create_encoder_for_testing(encoder_addr_1, 1 * SHANNONS_PER_SOMA), // 1 SOMA stake
        ];

        // Set encoder byte prices
        // The first encoder gets 45 as byte price
        encoders[0].byte_price = 45;
        encoders[0].next_epoch_byte_price = 45;

        // Initialize system state with validators and the first encoder
        let mut system_state = create_test_system_state(
            validators, encoders, 1000, // Supply amount
            10,   // Stake subsidy initial amount
        );

        // Verify validators were properly initialized
        assert_eq!(
            validator_stake_amount(&system_state, validator_addr_1).unwrap(),
            100 * SHANNONS_PER_SOMA
        );
        assert_eq!(
            validator_stake_amount(&system_state, validator_addr_2).unwrap(),
            150 * SHANNONS_PER_SOMA
        );

        // Verify initial reference byte price (only one encoder)
        // With only one encoder, its price becomes the reference price
        assert_eq!(system_state.encoders.reference_byte_price, 45);

        // Add some stake to the first encoder from a staker
        let staked_soma = stake_with_encoder(&mut system_state, staker_addr, encoder_addr_1, 4);

        // Advance epoch to activate stake
        advance_epoch_with_rewards(&mut system_state, 10 * SHANNONS_PER_SOMA).unwrap();

        // Verify the stake was added to the encoder
        assert_eq!(
            encoder_stake_amount(&system_state, encoder_addr_1).unwrap(),
            5 * SHANNONS_PER_SOMA // Original 1 + staked 4
        );

        // Create more encoders with different byte prices
        let mut encoder2 = create_encoder_for_testing(encoder_addr_2, 2 * SHANNONS_PER_SOMA);
        encoder2.byte_price = 42;
        encoder2.next_epoch_byte_price = 42;

        let mut encoder3 = create_encoder_for_testing(encoder_addr_3, 3 * SHANNONS_PER_SOMA);
        encoder3.byte_price = 40;
        encoder3.next_epoch_byte_price = 40;

        let mut encoder4 = create_encoder_for_testing(encoder_addr_4, 4 * SHANNONS_PER_SOMA);
        encoder4.byte_price = 41;
        encoder4.next_epoch_byte_price = 41;

        let mut encoder5 = create_encoder_for_testing(encoder_addr_5, 10 * SHANNONS_PER_SOMA);
        encoder5.byte_price = 43;
        encoder5.next_epoch_byte_price = 43;

        // Add the second encoder and advance epoch to activate it
        system_state
            .encoders
            .request_add_encoder(encoder2)
            .expect("Failed to add encoder 2");
        advance_epoch_with_rewards(&mut system_state, 10 * SHANNONS_PER_SOMA).unwrap();

        // Verify reference byte price with 2 encoders
        // With 2 encoders, voting power is distributed such that:
        // - Encoder 1 (price 45) + Encoder 2 (price 42) are needed to reach the threshold
        // - Since we process in descending price order, encoder 2 pushes us over the threshold
        // - Therefore, the reference price is 42
        assert_eq!(system_state.encoders.reference_byte_price, 42);

        // Add the third encoder and advance epoch
        system_state
            .encoders
            .request_add_encoder(encoder3)
            .expect("Failed to add encoder 3");
        advance_epoch_with_rewards(&mut system_state, 10 * SHANNONS_PER_SOMA).unwrap();

        // Verify reference byte price with 3 encoders
        // With 3 encoders, voting power is distributed such that:
        // - All 3 encoders are needed to reach the threshold
        // - Processing in price order: encoder 1 (45) + encoder 2 (42) + encoder 3 (40)
        // - Encoder 3 pushes us over the threshold
        // - Therefore, the reference price is 40
        assert_eq!(system_state.encoders.reference_byte_price, 40);

        // Add the fourth encoder and advance epoch
        system_state
            .encoders
            .request_add_encoder(encoder4)
            .expect("Failed to add encoder 4");
        advance_epoch_with_rewards(&mut system_state, 10 * SHANNONS_PER_SOMA).unwrap();

        // Verify reference byte price with 4 encoders
        // With 4 encoders, voting power is distributed such that:
        // - The threshold calculation requires multiple encoders
        // - Processing in price order (45, 42, 41, 40)
        // - Encoder 4 (price 41) is the one that pushes voting power over threshold
        // - Therefore, the reference price is 41
        assert_eq!(system_state.encoders.reference_byte_price, 41);

        // Add the fifth encoder and advance epoch
        system_state
            .encoders
            .request_add_encoder(encoder5)
            .expect("Failed to add encoder 5");
        advance_epoch_with_rewards(&mut system_state, 10 * SHANNONS_PER_SOMA).unwrap();

        // Verify reference byte price with 5 encoders
        // With 5 encoders, the voting power distribution still results in:
        // - The reference price is 41, which means adding encoder 5 didn't
        //   change the outcome despite its higher stake and price (43)
        // - This suggests the threshold was reached at encoder 4 which has price 41
        assert_eq!(system_state.encoders.reference_byte_price, 41);

        // Test setting a byte price through a transaction
        system_state
            .request_set_encoder_byte_price(encoder_addr_1, 50)
            .expect("Failed to set byte price");

        // Check that next_epoch_byte_price is updated but current byte_price remains the same
        let encoder1 = system_state.encoders.find_encoder(encoder_addr_1).unwrap();
        assert_eq!(encoder1.byte_price, 45);
        assert_eq!(encoder1.next_epoch_byte_price, 50);

        // Advance epoch to apply the new byte price
        advance_epoch_with_rewards(&mut system_state, 10 * SHANNONS_PER_SOMA).unwrap();

        // Verify that the byte price is updated
        let encoder1 = system_state.encoders.find_encoder(encoder_addr_1).unwrap();
        assert_eq!(encoder1.byte_price, 50);

        // Reference price remains at 41, showing that changing encoder 1's price
        // from 45 to 50 didn't affect the reference price calculation
        assert_eq!(system_state.encoders.reference_byte_price, 41);

        // Test the integration with epoch_start_state
        // The reference byte price should be available via EpochStartSystemState
        let epoch_start_state = system_state.into_epoch_start_state();
        assert_eq!(epoch_start_state.reference_byte_price, 41);
    }
}
