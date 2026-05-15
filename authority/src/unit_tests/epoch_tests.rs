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

use types::effects::ExecutionFailureStatus;
use types::system_state::{SystemState, SystemStateTrait as _};

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
async fn test_advance_epoch_routes_fees_to_protocol_fund() {
    // Under the new model, fees go to protocol_fund (not validators).
    // Validators are paid only via SOMA emissions.
    let mut state = get_genesis_system_state().await;
    let initial_fund = match &state {
        types::system_state::SystemState::V1(v1) => v1.protocol_fund_balance,
    };

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        state.protocol_version().into(),
        protocol_config::Chain::default(),
    );

    let fees = 1_000_000u64;
    // Use timestamp >= epoch_duration_ms so emissions are issued.
    let next_ts = state.epoch_start_timestamp_ms() + state.epoch_duration_ms();
    let result = state.advance_epoch(1, &protocol_config, fees, next_ts, vec![]);

    let validator_rewards = result.unwrap();

    // Validators receive rewards from emissions (not fees).
    assert!(
        !validator_rewards.is_empty(),
        "Validators should receive rewards from emissions"
    );

    // Fees should land in the protocol fund.
    let fund_after = match &state {
        types::system_state::SystemState::V1(v1) => v1.protocol_fund_balance,
    };
    assert_eq!(fund_after, initial_fund + fees, "Fees should accumulate in protocol_fund");
}

// =============================================================================
// Emission pool tests
// =============================================================================

#[tokio::test]
async fn test_advance_epoch_emission_pool_decreases() {
    let mut state = get_genesis_system_state().await;
    let initial_emission = state.emission_pool().balance;
    let emission_amount = state.emission_pool().current_distribution_amount;

    assert!(initial_emission > 0, "Genesis emission pool should be positive");
    assert!(emission_amount > 0, "Emission distribution amount should be positive");

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
    let initial_fund = match &state {
        types::system_state::SystemState::V1(v1) => v1.protocol_fund_balance,
    };
    let initial_emission_balance = state.emission_pool().balance;
    let initial_protocol_version = state.protocol_version();
    let fees = 5000u64;
    let timestamp = 2_000_000u64;

    // Call safe mode directly. Same protocol version (no upgrade in this test).
    state.advance_epoch_safe_mode(initial_epoch + 1, initial_protocol_version, fees, timestamp);

    assert_eq!(state.epoch(), initial_epoch + 1, "Epoch should advance in safe mode");
    assert!(state.safe_mode(), "Should be in safe mode");
    assert_eq!(state.epoch_start_timestamp_ms(), timestamp);

    // Fees should land in the protocol fund inline.
    let fund_after = match &state {
        types::system_state::SystemState::V1(v1) => v1.protocol_fund_balance,
    };
    assert_eq!(fund_after, initial_fund + fees, "Fees route to protocol_fund inline");

    // Emissions are forfeited — emission_pool is untouched.
    assert_eq!(
        state.emission_pool().balance,
        initial_emission_balance,
        "Emission pool balance untouched in safe mode"
    );
}

#[tokio::test]
async fn test_advance_epoch_safe_mode_accumulates_fees_across_epochs() {
    let mut state = get_genesis_system_state().await;
    let initial_fund = match &state {
        types::system_state::SystemState::V1(v1) => v1.protocol_fund_balance,
    };
    let pv = state.protocol_version();

    state.advance_epoch_safe_mode(1, pv, 1000, 1_000_000);
    state.advance_epoch_safe_mode(2, pv, 2000, 2_000_000);

    assert_eq!(state.epoch(), 2);
    let fund_after = match &state {
        types::system_state::SystemState::V1(v1) => v1.protocol_fund_balance,
    };
    assert_eq!(
        fund_after,
        initial_fund + 3000,
        "Fees accumulate in protocol_fund across safe-mode epochs"
    );
}

#[tokio::test]
async fn test_advance_epoch_recovery_from_safe_mode() {
    let mut state = get_genesis_system_state().await;
    let pv = state.protocol_version();

    // Enter safe mode
    state.advance_epoch_safe_mode(1, pv, 5000, 1_000_000);
    assert!(state.safe_mode());

    let protocol_config = protocol_config::ProtocolConfig::get_for_version(
        state.protocol_version().into(),
        protocol_config::Chain::default(),
    );

    // Recovery: successful advance_epoch should clear safe_mode flag.
    let result = state.advance_epoch(2, &protocol_config, 1000, 2_000_000, vec![]);
    assert!(result.is_ok(), "Recovery advance_epoch should succeed");

    assert!(!state.safe_mode(), "Should exit safe mode after successful advance");
}

#[tokio::test]
async fn test_advance_epoch_safe_mode_bumps_protocol_version() {
    // Safe mode must update protocol_version so a fix can land via upgrade
    // even while the chain is degraded. Matches Sui's behavior.
    let mut state = get_genesis_system_state().await;
    let initial_version = state.protocol_version();
    let new_version = initial_version + 1;

    state.advance_epoch_safe_mode(1, new_version, 0, 1_000_000);

    assert!(state.safe_mode());
    assert_eq!(state.protocol_version(), new_version, "protocol_version should bump in safe mode");
}

// Difficulty adjustment tests removed — target_state was stripped in Phase 1.

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
