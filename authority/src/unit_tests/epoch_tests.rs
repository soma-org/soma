// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests for epoch transition logic:
//! - SystemState::advance_epoch() arithmetic and state transitions
//! - Safe mode fallback behavior
//! - Emission pool allocation
//! - Difficulty adjustment
//!
//! These tests exercise advance_epoch() directly on SystemState objects
//! (without going through the full execution pipeline) to verify
//! arithmetic correctness in isolation.

use types::{
    effects::ExecutionFailureStatus,
    system_state::{SystemState, SystemStateTrait as _},
};

// =============================================================================
// Helper: get a default system state from a fresh authority
// =============================================================================

async fn get_genesis_system_state() -> SystemState {
    use crate::test_authority_builder::TestAuthorityBuilder;
    let authority_state = TestAuthorityBuilder::new().build().await;
    authority_state.get_system_state_object_for_testing().unwrap()
}

// =============================================================================
// advance_epoch basic tests
// =============================================================================

#[tokio::test]
async fn test_advance_epoch_basic() {
    let mut state = get_genesis_system_state().await;
    assert_eq!(state.epoch(), 0);

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        state.protocol_version().into(),
        protocol_config::Chain::default(),
    );

    let result = state.advance_epoch(
        1, // new_epoch
        &protocol_config,
        1000,      // fees collected
        1_000_000, // epoch_start_timestamp_ms
        vec![],    // epoch_randomness
    );

    assert!(result.is_ok(), "advance_epoch should succeed: {:?}", result.err());

    assert_eq!(state.epoch(), 1, "Epoch should be incremented to 1");
    assert!(state.epoch_start_timestamp_ms() >= 1_000_000, "Timestamp should be updated");
    assert!(!state.safe_mode(), "Should not be in safe mode after successful advance");
}

#[tokio::test]
async fn test_advance_epoch_wrong_epoch_rejected() {
    let mut state = get_genesis_system_state().await;
    assert_eq!(state.epoch(), 0);

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        state.protocol_version().into(),
        protocol_config::Chain::default(),
    );

    // Try to advance to epoch 5 (should be epoch 1)
    let result = state.advance_epoch(5, &protocol_config, 0, 0, vec![]);

    assert!(result.is_err(), "Should reject wrong epoch number");
    match result {
        Err(ExecutionFailureStatus::AdvancedToWrongEpoch) => {
            // Unit variant — epoch mismatch detected
        }
        Err(other) => panic!("Expected AdvancedToWrongEpoch, got {:?}", other),
        Ok(_) => panic!("Should have failed"),
    }
}

#[tokio::test]
async fn test_advance_epoch_returns_validator_rewards() {
    let mut state = get_genesis_system_state().await;

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        state.protocol_version().into(),
        protocol_config::Chain::default(),
    );

    // Advance with some fees collected
    let fees = 1_000_000u64;
    let result = state.advance_epoch(1, &protocol_config, fees, 1_000_000, vec![]);

    let validator_rewards = result.unwrap();
    // With non-zero fees + emissions, validators should receive some rewards
    // (unless validator_reward_allocation_bps is 0)
    if state.parameters().validator_reward_allocation_bps > 0 {
        assert!(
            !validator_rewards.is_empty(),
            "Validators should receive rewards when fees > 0 and allocation_bps > 0"
        );
    }
}

// =============================================================================
// Emission pool tests
// =============================================================================

#[tokio::test]
async fn test_advance_epoch_emission_pool_decreases() {
    let mut state = get_genesis_system_state().await;
    let initial_emission = state.emission_pool().balance;
    let emission_per_epoch = state.emission_pool().emission_per_epoch;

    assert!(initial_emission > 0, "Genesis emission pool should be positive");
    assert!(emission_per_epoch > 0, "Emission per epoch should be positive");

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        state.protocol_version().into(),
        protocol_config::Chain::default(),
    );

    // Timestamp must be >= prev_epoch_start + epoch_duration_ms to trigger emissions
    let future_timestamp =
        state.epoch_start_timestamp_ms() + state.parameters().epoch_duration_ms + 1;
    let _ = state.advance_epoch(1, &protocol_config, 0, future_timestamp, vec![]);

    // Emission pool should decrease by approximately emission_per_epoch
    // (some goes to rewards, some to targets)
    assert!(
        state.emission_pool().balance < initial_emission,
        "Emission pool should decrease after epoch: {} >= {}",
        state.emission_pool().balance,
        initial_emission
    );
}

// =============================================================================
// Safe mode tests
// =============================================================================

#[tokio::test]
async fn test_advance_epoch_safe_mode_basic() {
    let mut state = get_genesis_system_state().await;
    let initial_epoch = state.epoch();
    let fees = 5000u64;
    let timestamp = 2_000_000u64;

    // Call safe mode directly
    state.advance_epoch_safe_mode(initial_epoch + 1, fees, timestamp);

    assert_eq!(state.epoch(), initial_epoch + 1, "Epoch should still advance in safe mode");
    assert!(state.safe_mode(), "Should be in safe mode");
    assert_eq!(state.safe_mode_accumulated_fees(), fees, "Fees should accumulate in safe mode");
    assert!(
        state.safe_mode_accumulated_emissions() > 0,
        "Emissions should accumulate in safe mode (emission_per_epoch > 0)"
    );
    assert_eq!(
        state.epoch_start_timestamp_ms(),
        timestamp,
        "Timestamp should be updated in safe mode"
    );
}

#[tokio::test]
async fn test_advance_epoch_safe_mode_accumulates_across_epochs() {
    let mut state = get_genesis_system_state().await;

    // Enter safe mode for multiple epochs
    state.advance_epoch_safe_mode(1, 1000, 1_000_000);
    let first_fees = state.safe_mode_accumulated_fees();
    let first_emissions = state.safe_mode_accumulated_emissions();

    state.advance_epoch_safe_mode(2, 2000, 2_000_000);

    assert_eq!(state.epoch(), 2);
    assert_eq!(
        state.safe_mode_accumulated_fees(),
        first_fees + 2000,
        "Fees should accumulate across safe mode epochs"
    );
    assert!(
        state.safe_mode_accumulated_emissions() > first_emissions,
        "Emissions should accumulate across safe mode epochs"
    );
}

#[tokio::test]
async fn test_advance_epoch_recovery_from_safe_mode() {
    let mut state = get_genesis_system_state().await;

    // Enter safe mode
    state.advance_epoch_safe_mode(1, 5000, 1_000_000);
    assert!(state.safe_mode());
    assert_eq!(state.safe_mode_accumulated_fees(), 5000);

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        state.protocol_version().into(),
        protocol_config::Chain::default(),
    );

    // Recovery: successful advance_epoch should drain accumulators
    let result = state.advance_epoch(2, &protocol_config, 1000, 2_000_000, vec![]);
    assert!(result.is_ok(), "Recovery advance_epoch should succeed");

    assert!(!state.safe_mode(), "Should exit safe mode after successful advance");
    assert_eq!(
        state.safe_mode_accumulated_fees(),
        0,
        "Safe mode fees should be drained on recovery"
    );
    assert_eq!(
        state.safe_mode_accumulated_emissions(),
        0,
        "Safe mode emissions should be drained on recovery"
    );
}

// =============================================================================
// Difficulty adjustment tests
// =============================================================================

#[tokio::test]
async fn test_advance_epoch_hit_counter_tracking() {
    let mut state = get_genesis_system_state().await;

    // Simulate some hits and targets
    state.target_state_mut().record_target_generated();
    state.target_state_mut().record_target_generated();
    state.target_state_mut().record_hit();

    let targets_before = state.target_state().targets_generated_this_epoch;
    let hits_before = state.target_state().hits_this_epoch;
    assert_eq!(targets_before, 2);
    assert_eq!(hits_before, 1);

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        state.protocol_version().into(),
        protocol_config::Chain::default(),
    );

    let _ = state.advance_epoch(1, &protocol_config, 0, 1_000_000, vec![]);

    // After epoch advance, hit rate counters should be reset for the new epoch
    assert_eq!(state.target_state().hits_this_epoch, 0, "Hits should be reset after epoch advance");
    assert_eq!(
        state.target_state().targets_generated_this_epoch,
        0,
        "Targets generated should be reset after epoch advance"
    );
}

// =============================================================================
// BPS arithmetic safety
// =============================================================================

#[tokio::test]
async fn test_advance_epoch_u128_overflow_protection() {
    // Verify that large fee values don't cause u64 overflow in BPS calculations.
    // The implementation uses u128 intermediates: (value as u128 * bps as u128 / 10000) as u64
    let mut state = get_genesis_system_state().await;

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        state.protocol_version().into(),
        protocol_config::Chain::default(),
    );

    // Large fees that would overflow u64 multiplication without u128 intermediates
    let large_fees = u64::MAX / 2;
    let result = state.advance_epoch(1, &protocol_config, large_fees, 1_000_000, vec![]);

    // Should not panic due to overflow
    assert!(result.is_ok(), "Large fees should not cause overflow: {:?}", result.err());
}
