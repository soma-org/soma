#[cfg(test)]
#[allow(clippy::module_inception, clippy::unwrap_used, clippy::expect_used)]
mod model_tests {
    use crate::{
        base::{SomaAddress, dbg_addr},
        config::genesis_config::SHANNONS_PER_SOMA,
        effects::ExecutionFailureStatus,
        model::ModelId,
        object::ObjectID,
        system_state::{
            SystemState,
            staking::StakedSoma,
            test_utils::{
                self, ValidatorRewards, advance_epoch_with_reward_amounts,
                advance_epoch_with_rewards, commit_model, commit_model_update,
                commit_model_with_commission, create_test_system_state,
                create_validator_for_testing, reveal_model, reveal_model_update, stake_with_model,
                unstake,
            },
        },
    };

    // Deterministic addresses
    fn validator_addr_1() -> SomaAddress {
        dbg_addr(1)
    }
    fn validator_addr_2() -> SomaAddress {
        dbg_addr(2)
    }
    fn validator_addr_3() -> SomaAddress {
        dbg_addr(3)
    }
    fn model_owner() -> SomaAddress {
        dbg_addr(10)
    }
    fn delegator_addr() -> SomaAddress {
        dbg_addr(11)
    }

    fn model_id_1() -> ModelId {
        ObjectID::from_bytes([1u8; 32]).unwrap()
    }
    fn model_id_2() -> ModelId {
        ObjectID::from_bytes([2u8; 32]).unwrap()
    }

    /// Set up a system state with 2 validators (100 SOMA each) suitable for model tests.
    fn set_up_system_state() -> SystemState {
        let validators = vec![
            create_validator_for_testing(validator_addr_1(), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr_2(), 100 * SHANNONS_PER_SOMA),
        ];
        create_test_system_state(validators, 1000, 0)
    }

    /// Set up a system state with 3 validators for quorum tests.
    /// With 3 validators of equal stake, each gets ~3333 voting power.
    /// 2 out of 3 gives ~6666 which is just below QUORUM_THRESHOLD (6667).
    /// All 3 gives 10000 which exceeds threshold.
    /// To make 2-of-3 sufficient, we give validator 1 and 2 higher stake.
    fn set_up_system_state_for_reports() -> SystemState {
        // Give v1 and v2 enough combined stake to exceed quorum.
        // With 50/50/0 split, v1+v2 = 10000 (100%), exceeding 6667 threshold.
        // With 40/40/20, v1+v2 = ~8000, exceeding 6667.
        let validators = vec![
            create_validator_for_testing(validator_addr_1(), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr_2(), 100 * SHANNONS_PER_SOMA),
            create_validator_for_testing(validator_addr_3(), 50 * SHANNONS_PER_SOMA),
        ];
        create_test_system_state(validators, 1000, 0)
    }

    // ===================================================================
    // Commit-Reveal lifecycle
    // ===================================================================

    #[test]
    fn test_model_commit_reveal() {
        let mut state = set_up_system_state();

        // Epoch 0: Commit model
        let staked = commit_model(&mut state, model_owner(), model_id_1(), 5 * SHANNONS_PER_SOMA);

        // Model should be in pending_models
        assert!(state.model_registry.pending_models.contains_key(&model_id_1()));
        assert!(!state.model_registry.active_models.contains_key(&model_id_1()));

        // The model is committed but not revealed
        let model = state.model_registry.pending_models.get(&model_id_1()).unwrap();
        assert!(model.is_committed());
        assert!(!model.is_active());

        // Advance to epoch 1
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Epoch 1: Reveal model
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Model should now be in active_models
        assert!(!state.model_registry.pending_models.contains_key(&model_id_1()));
        assert!(state.model_registry.active_models.contains_key(&model_id_1()));

        let model = state.model_registry.active_models.get(&model_id_1()).unwrap();
        assert!(model.is_active());
        assert!(!model.is_committed());
        assert!(model.weights_manifest.is_some());
        assert_eq!(model.staking_pool.soma_balance, 5 * SHANNONS_PER_SOMA);

        // total_model_stake should reflect the active model
        assert_eq!(state.model_registry.total_model_stake, 5 * SHANNONS_PER_SOMA);
    }

    // ===================================================================
    // Commit without reveal -> slash
    // ===================================================================

    #[test]
    fn test_model_commit_no_reveal_slash() {
        let mut state = set_up_system_state();
        let stake = 10 * SHANNONS_PER_SOMA;

        // Epoch 0: Commit model
        let staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        assert!(state.model_registry.pending_models.contains_key(&model_id_1()));

        // Advance to epoch 1 (reveal window open, but we don't reveal)
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Model is still pending — reveal window is this epoch
        assert!(state.model_registry.pending_models.contains_key(&model_id_1()));

        // Advance to epoch 2 — the reveal window (epoch 1) has now passed
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Model should have been slashed and moved to inactive
        assert!(!state.model_registry.pending_models.contains_key(&model_id_1()));
        assert!(!state.model_registry.active_models.contains_key(&model_id_1()));
        assert!(state.model_registry.inactive_models.contains_key(&model_id_1()));

        let model = state.model_registry.inactive_models.get(&model_id_1()).unwrap();
        assert!(model.is_inactive());

        // Slash at 50% (model_reveal_slash_rate_bps = 5000)
        let expected_balance = stake - (stake * 5000 / 10000);
        assert_eq!(model.staking_pool.soma_balance, expected_balance);

        // total_model_stake should be 0 (model was never active)
        assert_eq!(state.model_registry.total_model_stake, 0);
    }

    // ===================================================================
    // Staking delegation
    // ===================================================================

    #[test]
    fn test_model_staking_delegation() {
        let mut state = set_up_system_state();
        let initial_stake = 10 * SHANNONS_PER_SOMA;

        // Epoch 0: Commit model
        let _owner_staked = commit_model(&mut state, model_owner(), model_id_1(), initial_stake);

        // Advance to epoch 1 and reveal
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Delegator stakes 20 SOMA to the model
        let delegator_staked = stake_with_model(&mut state, &model_id_1(), 20);

        // Advance epoch to activate delegation stake
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Total model stake should be initial + delegation
        let model = state.model_registry.active_models.get(&model_id_1()).unwrap();
        assert_eq!(model.staking_pool.soma_balance, initial_stake + 20 * SHANNONS_PER_SOMA);
        assert_eq!(state.model_registry.total_model_stake, initial_stake + 20 * SHANNONS_PER_SOMA);

        // Delegator withdraws
        let withdrawn = unstake(&mut state, delegator_staked);
        assert_eq!(withdrawn, 20 * SHANNONS_PER_SOMA);

        // Advance epoch to process withdrawal
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        let model = state.model_registry.active_models.get(&model_id_1()).unwrap();
        assert_eq!(model.staking_pool.soma_balance, initial_stake);
        assert_eq!(state.model_registry.total_model_stake, initial_stake);
    }

    // ===================================================================
    // Voluntary deactivation
    // ===================================================================

    #[test]
    fn test_model_deactivate_voluntary() {
        let mut state = set_up_system_state();
        let initial_stake = 10 * SHANNONS_PER_SOMA;

        // Commit + reveal
        let _owner_staked = commit_model(&mut state, model_owner(), model_id_1(), initial_stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Delegator stakes
        let delegator_staked = stake_with_model(&mut state, &model_id_1(), 5);

        // Advance to activate delegator stake
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Owner deactivates
        state.request_deactivate_model(model_owner(), &model_id_1()).expect("Failed to deactivate");

        assert!(!state.model_registry.active_models.contains_key(&model_id_1()));
        assert!(state.model_registry.inactive_models.contains_key(&model_id_1()));

        // No slash — balance should be intact
        let model = state.model_registry.inactive_models.get(&model_id_1()).unwrap();
        assert_eq!(model.staking_pool.soma_balance, initial_stake + 5 * SHANNONS_PER_SOMA);

        // total_model_stake should be 0
        assert_eq!(state.model_registry.total_model_stake, 0);

        // Delegator can still withdraw
        let withdrawn = unstake(&mut state, delegator_staked);
        assert_eq!(withdrawn, 5 * SHANNONS_PER_SOMA);
    }

    // ===================================================================
    // Minimum stake validation
    // ===================================================================

    #[test]
    fn test_model_min_stake_validation() {
        let mut state = set_up_system_state();

        // model_min_stake default is 1 SOMA = 1_000_000_000 shannons
        let min_stake = state.parameters.model_min_stake;

        // The executor validates min stake, not SystemState directly.
        // But we can verify that committing with the min stake works...
        let staked = commit_model(&mut state, model_owner(), model_id_1(), min_stake);
        assert!(state.model_registry.pending_models.contains_key(&model_id_1()));

        // ...and that commit_model itself accepts exactly model_min_stake
        assert_eq!(staked.principal, min_stake);
    }

    // ===================================================================
    // Architecture version validation
    // ===================================================================

    #[test]
    fn test_model_architecture_validation() {
        let mut state = set_up_system_state();
        let stake = 5 * SHANNONS_PER_SOMA;

        let url_str = format!("https://example.com/models/{}", model_id_1());
        let url_commitment = test_utils::url_commitment_for(&url_str);
        let weights_commitment = crate::digests::ModelWeightsCommitment::new([0xBB; 32]);
        let staking_pool_id = ObjectID::random();

        // Use wrong architecture version
        let wrong_version = state.parameters.model_architecture_version + 999;

        // request_commit_model does NOT validate architecture_version — the executor does.
        // The SystemState method just stores what it's given. So we verify the stored version.
        let result = state.request_commit_model(
            model_owner(),
            model_id_1(),
            url_commitment,
            weights_commitment,
            wrong_version,
            stake,
            0,
            staking_pool_id,
        );

        // The method succeeds (architecture check is in executor), but we verify the value
        assert!(result.is_ok());
        let model = state.model_registry.pending_models.get(&model_id_1()).unwrap();
        assert_eq!(model.architecture_version, wrong_version);
    }

    // ===================================================================
    // Model update commit-reveal
    // ===================================================================

    #[test]
    fn test_model_update_commit_reveal() {
        let mut state = set_up_system_state();
        let stake = 5 * SHANNONS_PER_SOMA;

        // Commit + reveal the model
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Save original weights_commitment
        let original_commitment =
            state.model_registry.active_models.get(&model_id_1()).unwrap().weights_commitment;

        // Commit model update (same epoch as reveal is fine)
        commit_model_update(&mut state, model_owner(), &model_id_1());

        let model = state.model_registry.active_models.get(&model_id_1()).unwrap();
        assert!(model.has_pending_update());

        // Advance to next epoch
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Reveal model update
        reveal_model_update(&mut state, model_owner(), &model_id_1());

        let model = state.model_registry.active_models.get(&model_id_1()).unwrap();
        assert!(!model.has_pending_update());

        // Weights commitment should have changed
        assert_ne!(model.weights_commitment, original_commitment);
        assert!(model.weights_manifest.is_some());
    }

    // ===================================================================
    // Model update no reveal -> cancel (no slash)
    // ===================================================================

    #[test]
    fn test_model_update_no_reveal_cancel() {
        let mut state = set_up_system_state();
        let stake = 5 * SHANNONS_PER_SOMA;

        // Commit + reveal
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Commit update
        commit_model_update(&mut state, model_owner(), &model_id_1());
        assert!(
            state.model_registry.active_models.get(&model_id_1()).unwrap().has_pending_update()
        );

        // Advance to reveal window epoch
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Don't reveal — advance again to expire the window
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Pending update should have been cleared (no slash)
        let model = state.model_registry.active_models.get(&model_id_1()).unwrap();
        assert!(!model.has_pending_update());
        assert!(model.is_active()); // Still active
        assert_eq!(model.staking_pool.soma_balance, stake); // No slash
    }

    // ===================================================================
    // Model update overwrite
    // ===================================================================

    #[test]
    fn test_model_update_overwrite() {
        let mut state = set_up_system_state();
        let stake = 5 * SHANNONS_PER_SOMA;

        // Commit + reveal
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // First commit update
        commit_model_update(&mut state, model_owner(), &model_id_1());
        let first_commitment = state
            .model_registry
            .active_models
            .get(&model_id_1())
            .unwrap()
            .pending_update
            .as_ref()
            .unwrap()
            .weights_commitment;

        // Second commit update in the same epoch — should overwrite
        let url_str2 = format!("https://example.com/models/{}/update-v2", model_id_1());
        let url_commitment2 = test_utils::url_commitment_for(&url_str2);
        let weights_commitment2 = crate::digests::ModelWeightsCommitment::new([0xDD; 32]);

        state
            .request_commit_model_update(
                model_owner(),
                &model_id_1(),
                url_commitment2,
                weights_commitment2,
            )
            .unwrap();

        let second_commitment = state
            .model_registry
            .active_models
            .get(&model_id_1())
            .unwrap()
            .pending_update
            .as_ref()
            .unwrap()
            .weights_commitment;

        // The second one should have overwritten the first
        assert_ne!(first_commitment, second_commitment);
        assert_eq!(second_commitment, weights_commitment2);
    }

    // ===================================================================
    // Commission rate
    // ===================================================================

    #[test]
    fn test_model_set_commission_rate() {
        let mut state = set_up_system_state();
        let stake = 5 * SHANNONS_PER_SOMA;

        // Commit + reveal
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Set commission rate for next epoch
        state.request_set_model_commission_rate(model_owner(), &model_id_1(), 1000).unwrap();

        // Current rate should still be 0
        let model = state.model_registry.active_models.get(&model_id_1()).unwrap();
        assert_eq!(model.commission_rate, 0);
        assert_eq!(model.next_epoch_commission_rate, 1000);

        // Advance epoch to apply
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        let model = state.model_registry.active_models.get(&model_id_1()).unwrap();
        assert_eq!(model.commission_rate, 1000);
        assert_eq!(model.next_epoch_commission_rate, 1000);
    }

    // ===================================================================
    // Commission rate too high
    // ===================================================================

    #[test]
    fn test_model_commission_rate_too_high() {
        let mut state = set_up_system_state();
        let stake = 5 * SHANNONS_PER_SOMA;

        // Commit with rate > BPS_DENOMINATOR should fail
        let url_str = format!("https://example.com/models/{}", model_id_1());
        let url_commitment = test_utils::url_commitment_for(&url_str);
        let weights_commitment = crate::digests::ModelWeightsCommitment::new([0xBB; 32]);
        let staking_pool_id = ObjectID::random();

        let result = state.request_commit_model(
            model_owner(),
            model_id_1(),
            url_commitment,
            weights_commitment,
            state.parameters.model_architecture_version,
            stake,
            10001, // > 10000
            staking_pool_id,
        );

        assert!(result.is_err());
        match result {
            Err(ExecutionFailureStatus::ModelCommissionRateTooHigh) => {}
            other => panic!("Expected ModelCommissionRateTooHigh, got {:?}", other),
        }

        // Also test set_model_commission_rate on an active model
        // First create one with valid rate
        let _staked = commit_model(&mut state, model_owner(), model_id_2(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_2());

        let result = state.request_set_model_commission_rate(model_owner(), &model_id_2(), 10001);
        assert!(result.is_err());
        match result {
            Err(ExecutionFailureStatus::ModelCommissionRateTooHigh) => {}
            other => panic!("Expected ModelCommissionRateTooHigh, got {:?}", other),
        }
    }

    // ===================================================================
    // Report quorum removal (2f+1 slash)
    // ===================================================================

    #[test]
    fn test_model_report_quorum_removal() {
        let mut state = set_up_system_state_for_reports();
        let stake = 10 * SHANNONS_PER_SOMA;

        // Commit + reveal
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Reports from validators 1 and 2 (combined voting power exceeds QUORUM_THRESHOLD)
        state.report_model(validator_addr_1(), &model_id_1()).unwrap();
        state.report_model(validator_addr_2(), &model_id_1()).unwrap();

        // Model is still active until epoch boundary
        assert!(state.model_registry.active_models.contains_key(&model_id_1()));

        // Advance epoch to trigger report processing
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Model should be slashed and moved to inactive
        assert!(!state.model_registry.active_models.contains_key(&model_id_1()));
        assert!(state.model_registry.inactive_models.contains_key(&model_id_1()));

        let model = state.model_registry.inactive_models.get(&model_id_1()).unwrap();
        assert!(model.is_inactive());

        // Slash at 95% (model_tally_slash_rate_bps = 9500)
        let expected_balance = stake - (stake * 9500 / 10000);
        assert_eq!(model.staking_pool.soma_balance, expected_balance);

        // Report records should be cleared
        assert!(state.model_registry.model_report_records.is_empty());

        // total_model_stake should be 0
        assert_eq!(state.model_registry.total_model_stake, 0);
    }

    // ===================================================================
    // Report below quorum (model remains active)
    // ===================================================================

    #[test]
    fn test_model_report_below_quorum() {
        let mut state = set_up_system_state_for_reports();
        let stake = 10 * SHANNONS_PER_SOMA;

        // Commit + reveal
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Only validator 3 reports (not enough voting power for quorum)
        state.report_model(validator_addr_3(), &model_id_1()).unwrap();

        // Advance epoch
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Model should still be active (report below quorum)
        assert!(state.model_registry.active_models.contains_key(&model_id_1()));
        assert!(!state.model_registry.inactive_models.contains_key(&model_id_1()));

        // Report records should be cleared regardless
        assert!(state.model_registry.model_report_records.is_empty());

        // Stake intact
        let model = state.model_registry.active_models.get(&model_id_1()).unwrap();
        assert_eq!(model.staking_pool.soma_balance, stake);
    }

    // ===================================================================
    // Report undo
    // ===================================================================

    #[test]
    fn test_model_report_undo() {
        let mut state = set_up_system_state_for_reports();
        let stake = 10 * SHANNONS_PER_SOMA;

        // Commit + reveal
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Report from validator 1
        state.report_model(validator_addr_1(), &model_id_1()).unwrap();
        assert!(state.model_registry.model_report_records.contains_key(&model_id_1()));

        // Undo the report
        state.undo_report_model(validator_addr_1(), &model_id_1()).unwrap();

        // Report record should be removed (empty set cleaned up)
        assert!(!state.model_registry.model_report_records.contains_key(&model_id_1()));

        // Trying to undo again should fail
        let result = state.undo_report_model(validator_addr_1(), &model_id_1());
        assert!(result.is_err());
        match result {
            Err(ExecutionFailureStatus::ReportRecordNotFound) => {}
            other => panic!("Expected ReportRecordNotFound, got {:?}", other),
        }
    }

    // ===================================================================
    // Withdraw stake from inactive model
    // ===================================================================

    #[test]
    fn test_withdraw_stake_from_inactive_model() {
        let mut state = set_up_system_state();
        let initial_stake = 10 * SHANNONS_PER_SOMA;

        // Commit + reveal
        let _owner_staked = commit_model(&mut state, model_owner(), model_id_1(), initial_stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Delegator stakes
        let delegator_staked = stake_with_model(&mut state, &model_id_1(), 15);

        // Advance to activate delegation
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Deactivate model
        state.request_deactivate_model(model_owner(), &model_id_1()).unwrap();

        // Delegator withdraws from inactive model
        let withdrawn = unstake(&mut state, delegator_staked);
        assert_eq!(withdrawn, 15 * SHANNONS_PER_SOMA);
    }

    // ===================================================================
    // Not-owner errors
    // ===================================================================

    #[test]
    fn test_model_not_owner_errors() {
        let mut state = set_up_system_state();
        let stake = 5 * SHANNONS_PER_SOMA;
        let not_owner = delegator_addr();

        // Commit + reveal
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);

        // Non-owner tries to reveal
        let url_str = format!("https://example.com/models/{}", model_id_1());
        let manifest = test_utils::make_weights_manifest(&url_str);
        let embedding = crate::tensor::SomaTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], vec![10]);
        let result = state.request_reveal_model(not_owner, &model_id_1(), manifest, embedding);
        match result {
            Err(ExecutionFailureStatus::NotModelOwner) => {}
            other => panic!("Expected NotModelOwner, got {:?}", other),
        }

        // Owner reveals correctly
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Non-owner tries to set commission
        let result = state.request_set_model_commission_rate(not_owner, &model_id_1(), 500);
        match result {
            Err(ExecutionFailureStatus::NotModelOwner) => {}
            other => panic!("Expected NotModelOwner, got {:?}", other),
        }

        // Non-owner tries to deactivate
        let result = state.request_deactivate_model(not_owner, &model_id_1());
        match result {
            Err(ExecutionFailureStatus::NotModelOwner) => {}
            other => panic!("Expected NotModelOwner, got {:?}", other),
        }

        // Non-owner tries to commit update
        let url_str2 = format!("https://example.com/models/{}/update", model_id_1());
        let url_commitment2 = test_utils::url_commitment_for(&url_str2);
        let wc2 = crate::digests::ModelWeightsCommitment::new([0xCC; 32]);
        let result =
            state.request_commit_model_update(not_owner, &model_id_1(), url_commitment2, wc2);
        match result {
            Err(ExecutionFailureStatus::NotModelOwner) => {}
            other => panic!("Expected NotModelOwner, got {:?}", other),
        }
    }

    // ===================================================================
    // Reveal epoch mismatch
    // ===================================================================

    #[test]
    fn test_model_reveal_epoch_mismatch() {
        let mut state = set_up_system_state();
        let stake = 5 * SHANNONS_PER_SOMA;

        // Epoch 0: Commit model
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);

        // Try to reveal in the same epoch (should fail — must be commit_epoch + 1)
        let url_str = format!("https://example.com/models/{}", model_id_1());
        let manifest = test_utils::make_weights_manifest(&url_str);
        let embedding = crate::tensor::SomaTensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], vec![10]);
        let result = state.request_reveal_model(model_owner(), &model_id_1(), manifest, embedding);

        match result {
            Err(ExecutionFailureStatus::ModelRevealEpochMismatch) => {}
            other => panic!("Expected ModelRevealEpochMismatch, got {:?}", other),
        }
    }

    // ===================================================================
    // Report non-validator
    // ===================================================================

    #[test]
    fn test_model_report_non_validator() {
        let mut state = set_up_system_state();
        let stake = 5 * SHANNONS_PER_SOMA;

        // Commit + reveal
        let _staked = commit_model(&mut state, model_owner(), model_id_1(), stake);
        let mut vr = ValidatorRewards::new(&state.validators.validators);
        advance_epoch_with_reward_amounts(&mut state, 0, &mut vr);
        reveal_model(&mut state, model_owner(), &model_id_1());

        // Non-validator tries to report
        let result = state.report_model(delegator_addr(), &model_id_1());
        match result {
            Err(ExecutionFailureStatus::NotAValidator) => {}
            other => panic!("Expected NotAValidator, got {:?}", other),
        }
    }
}
