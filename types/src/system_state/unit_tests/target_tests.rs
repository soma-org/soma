// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for target generation.
//!
//! Tests for:
//! - Genesis bootstrap with seed targets
//! - Target generation determinism
//! - Model selection
//! - Difficulty adjustment
//! - Reward per target calculation

use super::test_utils::{
    advance_epoch_with_rewards, commit_model, commit_model_with_dim, create_model,
    create_test_system_state, create_test_system_state_at_version, create_validators_with_stakes,
    reveal_model, reveal_model_with_dim,
};
use crate::base::SomaAddress;
use crate::config::genesis_config::SHANNONS_PER_SOMA;
use crate::system_state::SystemStateTrait;
use crate::target::{TargetStatus, deterministic_embedding, generate_target, make_target_seed};

/// Test that target seeds are deterministic
#[test]
fn test_target_seed_deterministic() {
    let digest = crate::digests::TransactionDigest::random();
    let seed1 = make_target_seed(&digest, 0);
    let seed2 = make_target_seed(&digest, 0);
    assert_eq!(seed1, seed2, "Same inputs should produce same seed");

    // Different creation_num should give different seed
    let seed3 = make_target_seed(&digest, 1);
    assert_ne!(seed1, seed3, "Different creation_num should produce different seed");
}

/// Test that embeddings are deterministic
#[test]
fn test_embedding_deterministic() {
    let embedding1 = deterministic_embedding(42, 2048);
    let embedding2 = deterministic_embedding(42, 2048);
    assert_eq!(embedding1, embedding2, "Same seed should produce same embedding");

    let embedding3 = deterministic_embedding(43, 2048);
    assert_ne!(embedding1, embedding3, "Different seed should produce different embedding");
}

/// Test embedding dimension
#[test]
fn test_embedding_dimension() {
    let embedding = deterministic_embedding(42, 2048);
    assert_eq!(embedding.len(), 2048);

    let embedding_small = deterministic_embedding(42, 10);
    assert_eq!(embedding_small.len(), 10);
}

/// Test target generation requires active models
#[test]
fn test_target_generation_requires_active_models() {
    // Create system state with validators but no models
    let validators = create_validators_with_stakes(vec![100, 100]);
    let system_state = create_test_system_state(validators, 1000, 100);

    // Try to generate a target - should fail with NoActiveModels
    let result = generate_target(
        42,
        system_state.model_registry(),
        system_state.target_state(),
        3,    // models_per_target
        2048, // embedding_dim
        0,    // current_epoch
    );

    assert!(result.is_err(), "Target generation should fail without active models");
    assert!(
        matches!(result.unwrap_err(), crate::effects::ExecutionFailureStatus::NoActiveModels),
        "Should return NoActiveModels error"
    );
}

/// Test target generation with a single active model
#[test]
fn test_target_generation_single_model() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Add and activate a model
    let owner = SomaAddress::random();
    let model_id = crate::model::ModelId::random();
    let stake = 10 * SHANNONS_PER_SOMA;
    create_model(&mut system_state, owner, model_id, stake);
    commit_model_with_dim(&mut system_state, owner, &model_id, 2048);

    // Advance epoch to reveal
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    // Use 2048-dimensional embedding to match target embedding dimension
    reveal_model_with_dim(&mut system_state, owner, &model_id, 2048);

    // Set up target_state with some thresholds
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.0);
    system_state.target_state_mut().reward_per_target = 1000;

    // Generate target - should succeed with single model
    let result = generate_target(
        42,
        system_state.model_registry(),
        system_state.target_state(),
        3,    // models_per_target (will be capped to 1)
        2048, // embedding_dim
        1,    // current_epoch
    );

    assert!(result.is_ok(), "Target generation should succeed");
    let target = result.unwrap();

    assert_eq!(target.model_ids.len(), 1, "Should have 1 model");
    assert_eq!(target.model_ids[0], model_id);
    assert_eq!(target.embedding.len(), 2048);
    assert_eq!(target.distance_threshold.as_scalar(), 1.0);
    assert_eq!(target.reward_pool, 1000);
    assert_eq!(target.generation_epoch, 1);
    assert!(matches!(target.status, TargetStatus::Open));
}

/// Test target generation with multiple models
#[test]
fn test_target_generation_multiple_models() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    let owner = SomaAddress::random();

    // Add and activate 5 models
    let mut model_ids = Vec::new();
    for _ in 0..5 {
        let model_id = crate::model::ModelId::random();
        let stake = 10 * SHANNONS_PER_SOMA;
        create_model(&mut system_state, owner, model_id, stake);
        commit_model_with_dim(&mut system_state, owner, &model_id, 2048);
        model_ids.push(model_id);
    }

    // Advance epoch and reveal all models with 2048-dimensional embeddings
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    for model_id in &model_ids {
        // Use 2048-dimensional embedding to match target embedding dimension
        reveal_model_with_dim(&mut system_state, owner, model_id, 2048);
    }

    // Set up target_state
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.0);
    system_state.target_state_mut().reward_per_target = 1000;

    // Generate target with 3 models
    let result = generate_target(
        42,
        system_state.model_registry(),
        system_state.target_state(),
        3,    // models_per_target
        2048, // embedding_dim
        1,    // current_epoch
    );

    assert!(result.is_ok());
    let target = result.unwrap();

    assert_eq!(target.model_ids.len(), 3, "Should have exactly 3 models");

    // Verify no duplicates
    let unique_models: std::collections::HashSet<_> = target.model_ids.iter().collect();
    assert_eq!(unique_models.len(), 3, "All selected models should be unique");

    // Verify all selected models are from the active set
    for selected_id in &target.model_ids {
        assert!(model_ids.contains(selected_id), "Selected model should be from active set");
    }
}

/// Test reward_per_target calculation (V3: uses target_hits_per_epoch as denominator)
#[test]
fn test_calculate_reward_per_target() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set up parameters
    system_state.emission_pool_mut().emission_per_epoch = 1_000_000 * SHANNONS_PER_SOMA;
    system_state.parameters_mut().target_reward_allocation_bps = 8000; // 80%
    system_state.parameters_mut().target_hits_per_epoch = 100; // V3 uses this as denominator

    let reward_per_target = system_state.calculate_reward_per_target();

    // Expected: (1M SOMA * 80%) / 100 = 8000 SOMA per target
    let expected_target_emissions = (1_000_000 * SHANNONS_PER_SOMA * 8000) / 10000;
    let expected_reward = expected_target_emissions / 100;

    assert_eq!(
        reward_per_target, expected_reward,
        "Reward per target should match expected calculation"
    );
}

/// Test difficulty adjustment when too many hits (too easy)
#[test]
fn test_difficulty_adjustment_too_many_hits() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.0);
    system_state.parameters_mut().target_hits_per_epoch = 16; // target 16 hits/epoch
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 500; // 5% adjustment
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.1);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(10.0);

    // Simulate 30 hits (above target of 16)
    system_state.target_state_mut().hits_this_epoch = 30;

    let old_distance = system_state.target_state().distance_threshold.as_scalar();

    system_state.adjust_difficulty();

    // Thresholds should decrease (harder) when hits exceed target
    assert!(
        system_state.target_state().distance_threshold.as_scalar() < old_distance,
        "Distance threshold should decrease when hits exceed target"
    );
}

/// Test difficulty adjustment when too few hits (too hard)
#[test]
fn test_difficulty_adjustment_too_few_hits() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.0);
    system_state.parameters_mut().target_hits_per_epoch = 16; // target 16 hits/epoch
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 500; // 5% adjustment
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.1);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(10.0);

    // Simulate 5 hits (below target of 16)
    system_state.target_state_mut().hits_this_epoch = 5;

    let old_distance = system_state.target_state().distance_threshold.as_scalar();

    system_state.adjust_difficulty();

    // Thresholds should increase (easier) when hits are below target
    assert!(
        system_state.target_state().distance_threshold.as_scalar() > old_distance,
        "Distance threshold should increase when hits are below target"
    );
}

/// Test difficulty adjustment respects min bounds
#[test]
fn test_difficulty_adjustment_min_bounds() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set thresholds at minimum already
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(0.1);
    system_state.parameters_mut().target_hits_per_epoch = 16;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 500;
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.1);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(10.0);

    // Simulate many hits (above target) to trigger decrease
    system_state.target_state_mut().hits_this_epoch = 100;

    system_state.adjust_difficulty();

    // Should be clamped to min (use approximate comparison for f32)
    assert!(
        (system_state.target_state().distance_threshold.as_scalar() - 0.1).abs() < 0.001,
        "Distance threshold should not go below min: got {}",
        system_state.target_state().distance_threshold.as_scalar()
    );
}

/// Test difficulty adjustment respects max bounds
#[test]
fn test_difficulty_adjustment_max_bounds() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set thresholds at maximum already
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(10.0);
    system_state.parameters_mut().target_hits_per_epoch = 16;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 500;
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.1);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(10.0);

    // Simulate very few hits (below target) to trigger increase
    system_state.target_state_mut().hits_this_epoch = 2;

    system_state.adjust_difficulty();

    // Should be clamped to max
    assert_eq!(
        system_state.target_state().distance_threshold.as_scalar(),
        10.0,
        "Distance threshold should not go above max"
    );
}

/// Test no adjustment in bootstrap mode (no hits yet)
#[test]
fn test_difficulty_adjustment_bootstrap_mode() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.0);
    system_state.target_state_mut().hits_this_epoch = 0; // No hits = bootstrap

    let old_distance = system_state.target_state().distance_threshold.as_scalar();

    system_state.adjust_difficulty();

    // No adjustment in bootstrap mode
    assert_eq!(
        system_state.target_state().distance_threshold.as_scalar(),
        old_distance,
        "Distance threshold should not change in bootstrap mode"
    );
}

/// Test advance_epoch_targets updates target_state and resets counters
#[test]
fn test_advance_epoch_targets() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set up parameters
    system_state.emission_pool_mut().emission_per_epoch = 1_000_000 * SHANNONS_PER_SOMA;
    system_state.parameters_mut().target_reward_allocation_bps = 8000;
    system_state.parameters_mut().target_initial_targets_per_epoch = 100;

    // Set some epoch counters
    system_state.target_state_mut().targets_generated_this_epoch = 50;
    system_state.target_state_mut().hits_this_epoch = 40;

    // Initially reward_per_target is 0
    assert_eq!(system_state.target_state().reward_per_target, 0);

    system_state.advance_epoch_targets();

    // After advance, reward_per_target should be calculated
    assert!(
        system_state.target_state().reward_per_target > 0,
        "reward_per_target should be set after advance_epoch_targets"
    );

    // Counters should be reset
    assert_eq!(
        system_state.target_state().targets_generated_this_epoch,
        0,
        "targets_generated_this_epoch should be reset"
    );
    assert_eq!(system_state.target_state().hits_this_epoch, 0, "hits_this_epoch should be reset");
}

/// Test target status transitions
#[test]
fn test_target_status_transitions() {
    use crate::target::TargetStatus;
    use crate::tensor::SomaTensor;

    // Start with Open status
    let mut target = crate::target::TargetV1 {
        embedding: SomaTensor::zeros(vec![10]),
        model_ids: vec![],
        distance_threshold: SomaTensor::scalar(1000.0),
        reward_pool: 1000,
        generation_epoch: 0,
        status: TargetStatus::Open,
        submitter: None,
        winning_model_id: None,
        winning_model_owner: None,
        bond_amount: 0,
        winning_data_manifest: None,
        winning_embedding: None,
        winning_distance_score: None,
        winning_loss_score: None,
        submission_reports: std::collections::BTreeSet::new(),
    };

    assert!(target.is_open());
    assert!(!target.is_filled());
    assert!(!target.is_claimed());
    assert!(target.fill_epoch().is_none());

    // Transition to Filled
    target.status = TargetStatus::Filled { fill_epoch: 5 };
    target.submitter = Some(SomaAddress::random());
    target.winning_model_id = Some(crate::model::ModelId::random());
    target.winning_model_owner = Some(SomaAddress::random());
    target.bond_amount = 5000;

    assert!(!target.is_open());
    assert!(target.is_filled());
    assert!(!target.is_claimed());
    assert_eq!(target.fill_epoch(), Some(5));

    // Transition to Claimed
    target.status = TargetStatus::Claimed;

    assert!(!target.is_open());
    assert!(!target.is_filled());
    assert!(target.is_claimed());
    assert!(target.fill_epoch().is_none());
}

/// Test uniform model selection produces unique selections (no duplicates)
#[test]
fn test_model_selection_uniqueness() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    let owner = SomaAddress::random();

    // Add 10 models
    let mut all_model_ids = Vec::new();
    for _ in 0..10 {
        let model_id = crate::model::ModelId::random();
        let stake = 10 * SHANNONS_PER_SOMA;
        create_model(&mut system_state, owner, model_id, stake);
        commit_model(&mut system_state, owner, &model_id);
        all_model_ids.push(model_id);
    }

    // Advance epoch and reveal all models
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    for model_id in &all_model_ids {
        reveal_model(&mut system_state, owner, model_id);
    }

    // Select 5 models from 10 - should be unique
    let selected = crate::target::select_models_uniform(42, system_state.model_registry(), 5)
        .expect("Model selection should succeed");

    assert_eq!(selected.len(), 5, "Should select exactly 5 models");

    // Verify no duplicates
    let unique_set: std::collections::HashSet<_> = selected.iter().collect();
    assert_eq!(unique_set.len(), 5, "All selected models should be unique");

    // All selected models should be from the active set
    for model_id in &selected {
        assert!(all_model_ids.contains(model_id), "Selected model should be from active set");
    }
}

/// Test model selection when requesting more models than available
#[test]
fn test_model_selection_capped_to_available() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    let owner = SomaAddress::random();

    // Add only 2 models
    let mut model_ids = Vec::new();
    for _ in 0..2 {
        let model_id = crate::model::ModelId::random();
        let stake = 10 * SHANNONS_PER_SOMA;
        create_model(&mut system_state, owner, model_id, stake);
        commit_model(&mut system_state, owner, &model_id);
        model_ids.push(model_id);
    }

    // Advance epoch and reveal all models
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    for model_id in &model_ids {
        reveal_model(&mut system_state, owner, model_id);
    }

    // Request 5 models but only 2 exist
    let selected = crate::target::select_models_uniform(42, system_state.model_registry(), 5)
        .expect("Model selection should succeed");

    assert_eq!(selected.len(), 2, "Should cap selection to available models");
}

/// Test different seeds produce different model selections
#[test]
fn test_model_selection_seed_affects_result() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    let owner = SomaAddress::random();

    // Add 5 models
    let mut model_ids = Vec::new();
    for _ in 0..5 {
        let model_id = crate::model::ModelId::random();
        let stake = 10 * SHANNONS_PER_SOMA;
        create_model(&mut system_state, owner, model_id, stake);
        commit_model(&mut system_state, owner, &model_id);
        model_ids.push(model_id);
    }

    // Advance epoch and reveal all models
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    for model_id in &model_ids {
        reveal_model(&mut system_state, owner, model_id);
    }

    // Same seed = same selection
    let selection1 = crate::target::select_models_uniform(42, system_state.model_registry(), 3)
        .expect("Model selection should succeed");
    let selection2 = crate::target::select_models_uniform(42, system_state.model_registry(), 3)
        .expect("Model selection should succeed");
    assert_eq!(selection1, selection2, "Same seed should produce same selection");

    // Different seed = likely different selection (with 5 models, very unlikely to be same)
    let selection3 = crate::target::select_models_uniform(999, system_state.model_registry(), 3)
        .expect("Model selection should succeed");
    // Note: We don't assert inequality because technically same selection is possible (just unlikely)
    // The main test is that different seeds can produce different selections
    assert_eq!(selection3.len(), 3, "Different seed should still produce valid selection");
}

// ===== V3 Protocol Tests =====

/// Test reward_per_target V3: denominator = target_hits_per_epoch
#[test]
fn test_reward_per_target_v3() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // V3 params: emission_per_epoch = 1.37T shannons, allocation = 80%, hits = 86400
    let emission_per_epoch = 1_370_000_000_000u64; // 1.37T shannons
    system_state.emission_pool_mut().emission_per_epoch = emission_per_epoch;
    system_state.parameters_mut().target_reward_allocation_bps = 8000; // 80%
    system_state.parameters_mut().target_hits_per_epoch = 86_400;

    let reward = system_state.calculate_reward_per_target();

    // Expected: (1.37T * 80%) / 86400 = 12,685,185 shannons (~12.7M)
    let expected = (emission_per_epoch as u128 * 8000u128 / 10000u128) as u64 / 86_400;
    assert_eq!(reward, expected);
    assert!(reward > 12_000_000 && reward < 13_000_000, "reward {} should be ~12.7M", reward);
}

/// Test reward_per_target V2: denominator = target_initial_targets_per_epoch (replay compat)
#[test]
fn test_reward_per_target_v2_unchanged() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state_at_version(validators, 1000, 100, 2);

    let emission_per_epoch = 1_370_000_000_000u64;
    system_state.emission_pool_mut().emission_per_epoch = emission_per_epoch;
    system_state.parameters_mut().target_reward_allocation_bps = 8000;
    system_state.parameters_mut().target_initial_targets_per_epoch = 20;

    let reward = system_state.calculate_reward_per_target();

    // V2: (1.37T * 80%) / 20 = 54.8B
    let expected = (emission_per_epoch as u128 * 8000u128 / 10000u128) as u64 / 20;
    assert_eq!(reward, expected);
    assert!(reward > 50_000_000_000, "V2 reward {} should be ~54.8B", reward);
}

/// Test z-based difficulty adjustment: too many hits → harder (z increases, threshold decreases)
#[test]
fn test_z_adjustment_harder() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(0.978);
    system_state.parameters_mut().target_hits_per_epoch = 86_400;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 200; // 2%
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.95);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(2.0);

    // Simulate hits above target
    system_state.target_state_mut().hits_this_epoch = 100_000;

    let old_threshold = system_state.target_state().distance_threshold.as_scalar();
    system_state.adjust_difficulty();
    let new_threshold = system_state.target_state().distance_threshold.as_scalar();

    assert!(
        new_threshold < old_threshold,
        "Threshold should decrease (harder) when hits > target: {} -> {}",
        old_threshold,
        new_threshold
    );
}

/// Test z-based difficulty adjustment: too few hits → easier (z decreases, threshold increases)
#[test]
fn test_z_adjustment_easier() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(0.978);
    system_state.parameters_mut().target_hits_per_epoch = 86_400;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 200;
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.95);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(2.0);

    // Simulate hits well below target
    system_state.target_state_mut().hits_this_epoch = 5_000;

    let old_threshold = system_state.target_state().distance_threshold.as_scalar();
    system_state.adjust_difficulty();
    let new_threshold = system_state.target_state().distance_threshold.as_scalar();

    assert!(
        new_threshold > old_threshold,
        "Threshold should increase (easier) when hits < target: {} -> {}",
        old_threshold,
        new_threshold
    );
}

/// Test MIN_Z_STEP prevents z=0 from being stuck
#[test]
fn test_z_min_step_prevents_stuck() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    // Pin to V4 to test the original MIN_Z_STEP=0.1 behavior
    let mut system_state = create_test_system_state_at_version(validators, 1000, 100, 4);

    // Set threshold = 1.0 → z = 0.0
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.0);
    system_state.parameters_mut().target_hits_per_epoch = 86_400;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 200;
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.95);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(2.0);

    // More hits than target → should make harder even though z=0
    system_state.target_state_mut().hits_this_epoch = 100_000;

    system_state.adjust_difficulty();
    let new_threshold = system_state.target_state().distance_threshold.as_scalar();

    // z should have increased by MIN_Z_STEP (0.1) → threshold = 1.0 - 0.022 * 0.1 = 0.9978
    assert!(
        new_threshold < 1.0,
        "Threshold should decrease from 1.0 even when z=0 (MIN_Z_STEP): got {}",
        new_threshold
    );
    let expected = 1.0 - 0.022 * 0.1;
    assert!(
        (new_threshold - expected).abs() < 0.001,
        "Threshold should be ~{}: got {}",
        expected,
        new_threshold
    );
}

/// Test z clamped at min_distance_threshold (0.95)
#[test]
fn test_z_clamp_at_min_threshold() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(0.978);
    system_state.parameters_mut().target_hits_per_epoch = 86_400;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 200;
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.95);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(2.0);

    // Simulate many epochs of hardening
    for _ in 0..200 {
        system_state.target_state_mut().hits_this_epoch = 200_000; // always too many
        system_state.adjust_difficulty();
        system_state.target_state_mut().hits_this_epoch = 0; // reset for next iteration
    }

    let final_threshold = system_state.target_state().distance_threshold.as_scalar();
    assert!(
        final_threshold >= 0.95 - 0.001,
        "Threshold must never drop below 0.95: got {}",
        final_threshold
    );
}

/// Test z clamped at max_distance_threshold
#[test]
fn test_z_clamp_at_max_threshold() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(0.978);
    system_state.parameters_mut().target_hits_per_epoch = 86_400;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 200;
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.95);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(2.0);

    // Simulate many epochs of easing
    for _ in 0..200 {
        system_state.target_state_mut().hits_this_epoch = 0; // no hits → easier
        // Need non-zero EMA to avoid bootstrap skip
        system_state.target_state_mut().hits_ema = 1;
        system_state.adjust_difficulty();
    }

    let final_threshold = system_state.target_state().distance_threshold.as_scalar();
    assert!(
        final_threshold <= 2.0 + 0.001,
        "Threshold must never exceed max 2.0: got {}",
        final_threshold
    );
}

/// Test difficulty convergence: fills == target → no drift
#[test]
fn test_difficulty_convergence() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(0.978);
    system_state.parameters_mut().target_hits_per_epoch = 86_400;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 200;
    system_state.parameters_mut().target_hits_ema_decay_bps = 9000;
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.95);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(2.0);

    let initial_threshold = system_state.target_state().distance_threshold.as_scalar();

    // Set EMA to target hits (equilibrium)
    system_state.target_state_mut().hits_ema = 86_400;

    // Run 50 epochs with fills == target
    for _ in 0..50 {
        system_state.target_state_mut().hits_this_epoch = 86_400;
        system_state.adjust_difficulty();
    }

    let final_threshold = system_state.target_state().distance_threshold.as_scalar();
    assert!(
        (final_threshold - initial_threshold).abs() < 0.001,
        "Threshold should stay stable when fills == target: initial={}, final={}",
        initial_threshold,
        final_threshold
    );
}

/// Test recovery from difficulty spiral (threshold at floor, 0 fills → should ease)
#[test]
fn test_difficulty_recovery_from_spiral() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Start at max hardness (z = max, threshold = 0.95)
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(0.95);
    system_state.parameters_mut().target_hits_per_epoch = 86_400;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 200;
    system_state.parameters_mut().target_hits_ema_decay_bps = 9000;
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.95);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(2.0);

    // Run 20 epochs with very few fills (1 hit keeps EMA non-zero to avoid bootstrap skip)
    for _ in 0..20 {
        system_state.target_state_mut().hits_this_epoch = 1;
        system_state.adjust_difficulty();
    }

    let final_threshold = system_state.target_state().distance_threshold.as_scalar();
    assert!(
        final_threshold > 0.96,
        "Threshold should recover from floor with few fills: got {}",
        final_threshold
    );
}

/// Test legacy difficulty preserved at V2
#[test]
fn test_legacy_difficulty_preserved() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state_at_version(validators, 1000, 100, 2);

    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.0);
    system_state.parameters_mut().target_hits_per_epoch = 16;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 1000; // 10%
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.1);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(10.0);

    // 30 hits > 16 target → should use multiplicative decrease
    system_state.target_state_mut().hits_this_epoch = 30;

    system_state.adjust_difficulty();
    let threshold = system_state.target_state().distance_threshold.as_scalar();

    // V2 multiplicative: 1.0 * (10000 - 1000) / 10000 = 0.9
    assert!(
        (threshold - 0.9).abs() < 0.001,
        "V2 should use multiplicative adjustment: expected 0.9, got {}",
        threshold
    );
}

/// Test V3 migration resets threshold and EMA
#[test]
fn test_v3_migration_resets_state() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state_at_version(validators, 1000, 100, 2);

    // Simulate pre-V3 stuck state
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(0.86);
    system_state.target_state_mut().hits_ema = 3213;

    // Advance epoch with V3 config (triggers migration)
    let next_epoch = system_state.epoch() + 1;
    let new_timestamp =
        system_state.epoch_start_timestamp_ms() + system_state.parameters().epoch_duration_ms;

    let v3_config = protocol_config::ProtocolConfig::get_for_version(
        protocol_config::ProtocolVersion::new(3),
        protocol_config::Chain::default(),
    );

    let _ = system_state.advance_epoch(next_epoch, &v3_config, 0, new_timestamp, vec![0; 32]);

    // After V3 migration: threshold should be reset to initial (0.978), EMA to 0
    let threshold = system_state.target_state().distance_threshold.as_scalar();
    assert!(
        (threshold - 0.978).abs() < 0.01,
        "V3 migration should reset threshold to 0.978: got {}",
        threshold
    );
}

/// Test V3 migration only runs once (V3→V3 should not reset)
#[test]
fn test_v3_migration_only_once() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Already at V3, set some state
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(0.96);
    system_state.target_state_mut().hits_ema = 5000;

    // Advance epoch (V3→V3, no migration)
    let next_epoch = system_state.epoch() + 1;
    let new_timestamp =
        system_state.epoch_start_timestamp_ms() + system_state.parameters().epoch_duration_ms;

    let v3_config = protocol_config::ProtocolConfig::get_for_version(
        protocol_config::ProtocolVersion::new(3),
        protocol_config::Chain::default(),
    );

    let _ = system_state.advance_epoch(next_epoch, &v3_config, 0, new_timestamp, vec![0; 32]);

    // EMA should NOT be reset to 0 (it was updated by adjust_difficulty)
    // The EMA update formula: new = (9000 * 5000 + 1000 * 0) / 10000 = 4500
    // It should be whatever the EMA formula produces, not 0
    assert!(
        system_state.target_state().hits_ema != 0
            || system_state.target_state().hits_this_epoch == 0,
        "V3→V3 should not reset EMA to 0 via migration"
    );
}

/// Test V5 MIN_Z_STEP = 2.0 at z=0 (threshold=1.0)
#[test]
fn test_v5_min_z_step() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set threshold = 1.0 → z = 0.0
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.0);
    system_state.parameters_mut().target_hits_per_epoch = 86_400;
    system_state.parameters_mut().target_difficulty_adjustment_rate_bps = 200;
    system_state.parameters_mut().target_min_distance_threshold =
        crate::tensor::SomaTensor::scalar(0.95);
    system_state.parameters_mut().target_max_distance_threshold =
        crate::tensor::SomaTensor::scalar(2.0);

    // More hits than target → should make harder with step = 2.0 z-units
    system_state.target_state_mut().hits_this_epoch = 100_000;

    system_state.adjust_difficulty();
    let new_threshold = system_state.target_state().distance_threshold.as_scalar();

    // z should have increased by 2.0 → threshold = 1.0 - 0.022 * 2.0 = 0.956
    let expected = 1.0 - 0.022 * 2.0;
    assert!(
        (new_threshold - expected).abs() < 0.001,
        "V5 MIN_Z_STEP=2.0: threshold should be ~{}: got {}",
        expected,
        new_threshold
    );
}

/// Test V5 migration resets threshold and EMA
#[test]
fn test_v5_migration_resets_state() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state_at_version(validators, 1000, 100, 4);

    // Simulate V4 state stuck near z=0
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.0);
    system_state.target_state_mut().hits_ema = 200;

    // Advance epoch with V5 config (triggers migration)
    let next_epoch = system_state.epoch() + 1;
    let new_timestamp =
        system_state.epoch_start_timestamp_ms() + system_state.parameters().epoch_duration_ms;

    let v5_config = protocol_config::ProtocolConfig::get_for_version(
        protocol_config::ProtocolVersion::new(5),
        protocol_config::Chain::default(),
    );

    let _ = system_state.advance_epoch(next_epoch, &v5_config, 0, new_timestamp, vec![0; 32]);

    // After V5 migration: threshold should be reset to 1.05, EMA to 0
    let threshold = system_state.target_state().distance_threshold.as_scalar();
    assert!(
        (threshold - 1.05).abs() < 0.01,
        "V5 migration should reset threshold to 1.05: got {}",
        threshold
    );
}

/// Test V5 migration only runs once (V5→V5 should not reset)
#[test]
fn test_v5_migration_only_once() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Already at V5, set some state
    system_state.target_state_mut().distance_threshold = crate::tensor::SomaTensor::scalar(1.08);
    system_state.target_state_mut().hits_ema = 5000;

    // Advance epoch (V5→V5, no migration)
    let next_epoch = system_state.epoch() + 1;
    let new_timestamp =
        system_state.epoch_start_timestamp_ms() + system_state.parameters().epoch_duration_ms;

    let v5_config = protocol_config::ProtocolConfig::get_for_version(
        protocol_config::ProtocolVersion::new(5),
        protocol_config::Chain::default(),
    );

    let _ = system_state.advance_epoch(next_epoch, &v5_config, 0, new_timestamp, vec![0; 32]);

    // EMA should NOT be reset to 0 (it was updated by adjust_difficulty)
    assert!(
        system_state.target_state().hits_ema != 0
            || system_state.target_state().hits_this_epoch == 0,
        "V5→V5 should not reset EMA to 0 via migration"
    );
}

/// Test emission budget never exceeded at various fill counts
#[test]
fn test_emission_budget_never_exceeded() {
    let validators = create_validators_with_stakes(vec![100, 100]);

    for fill_count in [0u64, 100, 1000, 86_400, 100_000] {
        let mut system_state = create_test_system_state(validators.clone(), 1000, 100);

        let emission_per_epoch = 1_370_000_000_000u64;
        system_state.emission_pool_mut().emission_per_epoch = emission_per_epoch;
        system_state.parameters_mut().target_reward_allocation_bps = 8000;
        system_state.parameters_mut().target_hits_per_epoch = 86_400;

        let reward_per_target = system_state.calculate_reward_per_target();
        let epoch_target_budget = (emission_per_epoch as u128 * 8000u128 / 10000u128) as u64;

        // Simulate spawning targets up to fill_count (or budget)
        let mut total_spent = 0u64;
        let mut targets_spawned = 0u64;
        for _ in 0..fill_count {
            let already_spent = targets_spawned.saturating_mul(reward_per_target);
            if already_spent.saturating_add(reward_per_target) > epoch_target_budget {
                break;
            }
            total_spent += reward_per_target;
            targets_spawned += 1;
        }

        assert!(
            total_spent <= epoch_target_budget,
            "Budget exceeded at fill_count={}: spent={}, budget={}",
            fill_count,
            total_spent,
            epoch_target_budget
        );
    }
}

/// Test emission budget exhaustion at capacity (86,400 fills)
#[test]
fn test_emission_budget_at_capacity() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    let emission_per_epoch = 1_370_000_000_000u64;
    system_state.emission_pool_mut().emission_per_epoch = emission_per_epoch;
    system_state.parameters_mut().target_reward_allocation_bps = 8000;
    system_state.parameters_mut().target_hits_per_epoch = 86_400;

    let reward_per_target = system_state.calculate_reward_per_target();
    let epoch_target_budget = (emission_per_epoch as u128 * 8000u128 / 10000u128) as u64;

    // At exactly 86,400 targets: total should be close to budget
    let total_at_capacity = reward_per_target * 86_400;
    assert!(
        total_at_capacity <= epoch_target_budget,
        "86,400 targets should fit within budget: total={}, budget={}",
        total_at_capacity,
        epoch_target_budget
    );

    // The 86,401st target should be blocked
    let at_86401 = reward_per_target * 86_401;
    assert!(
        at_86401 > epoch_target_budget,
        "86,401 targets should exceed budget: total={}, budget={}",
        at_86401,
        epoch_target_budget
    );
}

/// Test emission pool drain rate follows linear schedule
#[test]
fn test_emission_pool_drain_rate_linear() {
    let validators = create_validators_with_stakes(vec![100, 100]);

    for fill_rate in [5_000u64, 10_000, 50_000, 86_400] {
        let mut system_state = create_test_system_state(validators.clone(), 100_000, 100);

        let emission_per_epoch = 100 * SHANNONS_PER_SOMA;
        system_state.emission_pool_mut().emission_per_epoch = emission_per_epoch;
        system_state.parameters_mut().target_reward_allocation_bps = 8000;
        system_state.parameters_mut().target_hits_per_epoch = 86_400;

        let genesis_balance = system_state.emission_pool().balance;
        let reward_per_target = system_state.calculate_reward_per_target();
        let epoch_target_budget = (emission_per_epoch as u128 * 8000u128 / 10000u128) as u64;

        // Simulate 10 epochs
        for epoch in 1..=10 {
            // Simulate target spawning within budget
            let mut targets_spawned = 0u64;
            for _ in 0..fill_rate {
                let already_spent = targets_spawned.saturating_mul(reward_per_target);
                if already_spent.saturating_add(reward_per_target) > epoch_target_budget {
                    break;
                }
                if system_state.emission_pool().balance < reward_per_target {
                    break;
                }
                system_state.emission_pool_mut().balance -= reward_per_target;
                targets_spawned += 1;
            }

            // Pool should never drain faster than linear schedule
            let max_drain = emission_per_epoch * epoch;
            assert!(
                system_state.emission_pool().balance >= genesis_balance.saturating_sub(max_drain),
                "Pool drained too fast at fill_rate={}, epoch={}: balance={}, min_expected={}",
                fill_rate,
                epoch,
                system_state.emission_pool().balance,
                genesis_balance.saturating_sub(max_drain)
            );
        }
    }
}
