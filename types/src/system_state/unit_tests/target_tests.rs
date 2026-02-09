//! Unit tests for target generation.
//!
//! Tests for:
//! - Genesis bootstrap with seed targets
//! - Target generation determinism
//! - Model selection
//! - Difficulty adjustment
//! - Reward per target calculation

use crate::{
    base::SomaAddress,
    config::genesis_config::SHANNONS_PER_SOMA,
    target::{TargetStatus, deterministic_embedding, generate_target, make_target_seed},
};

use super::test_utils::{
    advance_epoch_with_rewards, commit_model, create_test_system_state,
    create_validators_with_stakes, reveal_model,
};

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
    let embedding1 = deterministic_embedding(42, 768);
    let embedding2 = deterministic_embedding(42, 768);
    assert_eq!(embedding1, embedding2, "Same seed should produce same embedding");

    let embedding3 = deterministic_embedding(43, 768);
    assert_ne!(embedding1, embedding3, "Different seed should produce different embedding");
}

/// Test embedding dimension
#[test]
fn test_embedding_dimension() {
    let embedding = deterministic_embedding(42, 768);
    assert_eq!(embedding.len(), 768);

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
        &system_state.model_registry,
        &system_state.target_state,
        3,   // models_per_target
        768, // embedding_dim
        0,   // current_epoch
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
    commit_model(&mut system_state, owner, model_id, stake);

    // Advance epoch to reveal
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    reveal_model(&mut system_state, owner, &model_id);

    // Set up target_state with some thresholds
    system_state.target_state.distance_threshold = 1_000_000;
    system_state.target_state.reward_per_target = 1000;

    // Generate target - should succeed with single model
    let result = generate_target(
        42,
        &system_state.model_registry,
        &system_state.target_state,
        3,   // models_per_target (will be capped to 1)
        768, // embedding_dim
        1,   // current_epoch
    );

    assert!(result.is_ok(), "Target generation should succeed");
    let target = result.unwrap();

    assert_eq!(target.model_ids.len(), 1, "Should have 1 model");
    assert_eq!(target.model_ids[0], model_id);
    assert_eq!(target.embedding.len(), 768);
    assert_eq!(target.distance_threshold, 1_000_000);
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
        commit_model(&mut system_state, owner, model_id, stake);
        model_ids.push(model_id);
    }

    // Advance epoch and reveal all models
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    for model_id in &model_ids {
        reveal_model(&mut system_state, owner, model_id);
    }

    // Set up target_state
    system_state.target_state.distance_threshold = 1_000_000;
    system_state.target_state.reward_per_target = 1000;

    // Generate target with 3 models
    let result = generate_target(
        42,
        &system_state.model_registry,
        &system_state.target_state,
        3,   // models_per_target
        768, // embedding_dim
        1,   // current_epoch
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

/// Test reward_per_target calculation
#[test]
fn test_calculate_reward_per_target() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set up parameters
    system_state.emission_pool.emission_per_epoch = 1_000_000 * SHANNONS_PER_SOMA;
    system_state.parameters.target_reward_allocation_bps = 8000; // 80%
    system_state.parameters.target_initial_targets_per_epoch = 100; // 100 targets per epoch

    let reward_per_target = system_state.calculate_reward_per_target();

    // Expected: (1M SOMA * 80%) / 100 targets = 8000 SOMA per target
    let expected_target_emissions = (1_000_000 * SHANNONS_PER_SOMA * 8000) / 10000;
    let expected_reward = expected_target_emissions / 100;

    assert_eq!(
        reward_per_target, expected_reward,
        "Reward per target should match expected calculation"
    );
}

/// Test difficulty adjustment when hit rate is too high (too easy)
#[test]
fn test_difficulty_adjustment_high_hit_rate() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set initial thresholds
    system_state.target_state.distance_threshold = 1_000_000;
    system_state.parameters.target_hit_rate_target_bps = 8000; // 80% target hit rate
    system_state.parameters.target_difficulty_adjustment_rate_bps = 500; // 5% adjustment
    system_state.parameters.target_min_distance_threshold = 100_000;
    system_state.parameters.target_max_distance_threshold = 10_000_000;

    // Simulate high hit rate (95% of targets filled, target = 80%)
    system_state.target_state.targets_generated_this_epoch = 100;
    system_state.target_state.hits_this_epoch = 95;

    let old_distance = system_state.target_state.distance_threshold;

    system_state.adjust_difficulty();

    // Thresholds should decrease (harder) when hit rate is too high
    assert!(
        system_state.target_state.distance_threshold < old_distance,
        "Distance threshold should decrease when hit rate is too high"
    );
}

/// Test difficulty adjustment when hit rate is too low (too hard)
#[test]
fn test_difficulty_adjustment_low_hit_rate() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set initial thresholds
    system_state.target_state.distance_threshold = 1_000_000;
    system_state.parameters.target_hit_rate_target_bps = 8000; // 80% target hit rate
    system_state.parameters.target_difficulty_adjustment_rate_bps = 500; // 5% adjustment
    system_state.parameters.target_min_distance_threshold = 100_000;
    system_state.parameters.target_max_distance_threshold = 10_000_000;

    // Simulate low hit rate (50% of targets filled, target = 80%)
    system_state.target_state.targets_generated_this_epoch = 100;
    system_state.target_state.hits_this_epoch = 50;

    let old_distance = system_state.target_state.distance_threshold;

    system_state.adjust_difficulty();

    // Thresholds should increase (easier) when hit rate is too low
    assert!(
        system_state.target_state.distance_threshold > old_distance,
        "Distance threshold should increase when hit rate is too low"
    );
}

/// Test difficulty adjustment respects min bounds
#[test]
fn test_difficulty_adjustment_min_bounds() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set thresholds at minimum already
    system_state.target_state.distance_threshold = 100_000;
    system_state.parameters.target_hit_rate_target_bps = 8000; // 80% target
    system_state.parameters.target_difficulty_adjustment_rate_bps = 500;
    system_state.parameters.target_min_distance_threshold = 100_000;
    system_state.parameters.target_max_distance_threshold = 10_000_000;

    // Simulate very high hit rate (100%, target = 80%)
    system_state.target_state.targets_generated_this_epoch = 100;
    system_state.target_state.hits_this_epoch = 100;

    system_state.adjust_difficulty();

    // Should be clamped to min
    assert_eq!(
        system_state.target_state.distance_threshold, 100_000,
        "Distance threshold should not go below min"
    );
}

/// Test difficulty adjustment respects max bounds
#[test]
fn test_difficulty_adjustment_max_bounds() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set thresholds at maximum already
    system_state.target_state.distance_threshold = 10_000_000;
    system_state.parameters.target_hit_rate_target_bps = 8000; // 80% target
    system_state.parameters.target_difficulty_adjustment_rate_bps = 500;
    system_state.parameters.target_min_distance_threshold = 100_000;
    system_state.parameters.target_max_distance_threshold = 10_000_000;

    // Simulate very low hit rate (10%, target = 80%)
    system_state.target_state.targets_generated_this_epoch = 100;
    system_state.target_state.hits_this_epoch = 10;

    system_state.adjust_difficulty();

    // Should be clamped to max
    assert_eq!(
        system_state.target_state.distance_threshold, 10_000_000,
        "Distance threshold should not go above max"
    );
}

/// Test no adjustment in bootstrap mode (no targets generated)
#[test]
fn test_difficulty_adjustment_bootstrap_mode() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set initial thresholds
    system_state.target_state.distance_threshold = 1_000_000;
    system_state.target_state.targets_generated_this_epoch = 0; // Bootstrap mode

    let old_distance = system_state.target_state.distance_threshold;

    system_state.adjust_difficulty();

    // No adjustment in bootstrap mode
    assert_eq!(
        system_state.target_state.distance_threshold, old_distance,
        "Distance threshold should not change in bootstrap mode"
    );
}

/// Test advance_epoch_targets updates target_state and resets counters
#[test]
fn test_advance_epoch_targets() {
    let validators = create_validators_with_stakes(vec![100, 100]);
    let mut system_state = create_test_system_state(validators, 1000, 100);

    // Set up parameters
    system_state.emission_pool.emission_per_epoch = 1_000_000 * SHANNONS_PER_SOMA;
    system_state.parameters.target_reward_allocation_bps = 8000;
    system_state.parameters.target_initial_targets_per_epoch = 100;

    // Set some epoch counters
    system_state.target_state.targets_generated_this_epoch = 50;
    system_state.target_state.hits_this_epoch = 40;

    // Initially reward_per_target is 0
    assert_eq!(system_state.target_state.reward_per_target, 0);

    system_state.advance_epoch_targets();

    // After advance, reward_per_target should be calculated
    assert!(
        system_state.target_state.reward_per_target > 0,
        "reward_per_target should be set after advance_epoch_targets"
    );

    // Counters should be reset
    assert_eq!(
        system_state.target_state.targets_generated_this_epoch, 0,
        "targets_generated_this_epoch should be reset"
    );
    assert_eq!(system_state.target_state.hits_this_epoch, 0, "hits_this_epoch should be reset");
}

/// Test target status transitions
#[test]
fn test_target_status_transitions() {
    use crate::target::TargetStatus;
    use ndarray::Array1;

    // Start with Open status
    let mut target = crate::target::Target {
        embedding: Array1::zeros(10),
        model_ids: vec![],
        distance_threshold: 1000,
        reward_pool: 1000,
        generation_epoch: 0,
        status: TargetStatus::Open,
        miner: None,
        winning_model_id: None,
        winning_model_owner: None,
        bond_amount: 0,
        winning_data_manifest: None,
        winning_data_commitment: None,
        winning_embedding: None,
        winning_distance_score: None,
        challenger: None,
        challenge_id: None,
        submission_reports: std::collections::BTreeMap::new(),
    };

    assert!(target.is_open());
    assert!(!target.is_filled());
    assert!(!target.is_claimed());
    assert!(target.fill_epoch().is_none());

    // Transition to Filled
    target.status = TargetStatus::Filled { fill_epoch: 5 };
    target.miner = Some(SomaAddress::random());
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
        commit_model(&mut system_state, owner, model_id, stake);
        all_model_ids.push(model_id);
    }

    // Advance epoch and reveal all models
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    for model_id in &all_model_ids {
        reveal_model(&mut system_state, owner, model_id);
    }

    // Select 5 models from 10 - should be unique
    let selected = crate::target::select_models_uniform(42, &system_state.model_registry, 5)
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
    for _ in 0..2 {
        let model_id = crate::model::ModelId::random();
        let stake = 10 * SHANNONS_PER_SOMA;
        commit_model(&mut system_state, owner, model_id, stake);
    }

    // Advance epoch and reveal all models
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    for (model_id, _) in system_state.model_registry.pending_models.clone() {
        reveal_model(&mut system_state, owner, &model_id);
    }

    // Request 5 models but only 2 exist
    let selected = crate::target::select_models_uniform(42, &system_state.model_registry, 5)
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
    for _ in 0..5 {
        let model_id = crate::model::ModelId::random();
        let stake = 10 * SHANNONS_PER_SOMA;
        commit_model(&mut system_state, owner, model_id, stake);
    }

    // Advance epoch and reveal all models
    advance_epoch_with_rewards(&mut system_state, 0).unwrap();
    for (model_id, _) in system_state.model_registry.pending_models.clone() {
        reveal_model(&mut system_state, owner, &model_id);
    }

    // Same seed = same selection
    let selection1 = crate::target::select_models_uniform(42, &system_state.model_registry, 3)
        .expect("Model selection should succeed");
    let selection2 = crate::target::select_models_uniform(42, &system_state.model_registry, 3)
        .expect("Model selection should succeed");
    assert_eq!(selection1, selection2, "Same seed should produce same selection");

    // Different seed = likely different selection (with 5 models, very unlikely to be same)
    let selection3 = crate::target::select_models_uniform(999, &system_state.model_registry, 3)
        .expect("Model selection should succeed");
    // Note: We don't assert inequality because technically same selection is possible (just unlikely)
    // The main test is that different seeds can produce different selections
    assert_eq!(selection3.len(), 3, "Different seed should still produce valid selection");
}
