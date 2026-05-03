// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tests for validator management transactions:
//! SetCommissionRate, UpdateValidatorMetadata, ReportValidator, UndoReportValidator.
//!
//! These tests exercise the ValidatorExecutor through the full authority pipeline.
//! Default ConfigBuilder creates a single-validator committee, which means
//! certify_transaction gets quorum from one vote (100% of stake).

use std::sync::Arc;

use fastcrypto::ed25519::Ed25519KeyPair;
use types::SYSTEM_STATE_OBJECT_ID;
use types::base::SomaAddress;
use types::config::network_config::ConfigBuilder;
use types::crypto::{SomaKeyPair, get_key_pair};
use types::effects::{ExecutionStatus, TransactionEffectsAPI};
use types::object::ObjectID;
use types::transaction::{TransactionData, TransactionKind, UpdateValidatorMetadataArgs};
use types::unit_tests::utils::to_sender_signed_transaction;

use crate::authority::AuthorityState;
use crate::authority_test_utils::send_and_confirm_transaction_;
use crate::test_authority_builder::TestAuthorityBuilder;

// =============================================================================
// Helper: build authority state with access to validator keys
// =============================================================================

struct ValidatorTestSetup {
    authority_state: Arc<AuthorityState>,
    /// Address of validator 0 (signer for most tests)
    v0_address: SomaAddress,
}

async fn setup_validator_test() -> (ValidatorTestSetup, SomaKeyPair) {
    // Default ConfigBuilder creates a single-validator committee.
    // With 1 validator, certify_transaction gets quorum from one vote (100% stake).
    let network_config = ConfigBuilder::new_with_temp_dir().build();

    let v0_config = &network_config.validator_configs()[0];
    let v0_address = v0_config.soma_address();
    let v0_key = v0_config.account_key_pair.keypair().copy();

    let authority_state =
        TestAuthorityBuilder::new().with_network_config(&network_config, 0).build().await;

    // Stage 13c: seed USDC accumulator for balance-mode gas.
    authority_state
        .database_for_testing()
        .set_balance(v0_address, types::object::CoinType::Usdc, 50_000_000)
        .unwrap();

    (ValidatorTestSetup { authority_state, v0_address }, v0_key)
}

// =============================================================================
// SetCommissionRate tests
// =============================================================================

#[tokio::test]
async fn test_set_commission_rate_success() {
    let (setup, v0_key) = setup_validator_test().await;

    let new_rate = 500u64; // 5% commission

    let data = TransactionData::new(
        TransactionKind::SetCommissionRate { new_rate },
        setup.v0_address,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &v0_key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    assert!(result.is_ok(), "SetCommissionRate should succeed: {:?}", result.err());
    let (_, effects) = result.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);

    // SystemState should be mutated
    let mutated_ids: Vec<ObjectID> = effects.mutated().iter().map(|m| m.0.0).collect();
    assert!(
        mutated_ids.contains(&SYSTEM_STATE_OBJECT_ID),
        "SystemState should be mutated during SetCommissionRate"
    );

    // Verify commission rate was set (takes effect next epoch)
    let system_state = setup.authority_state.get_system_state_object_for_testing().unwrap();
    let validator = system_state
        .validators()
        .validators
        .iter()
        .find(|v| v.metadata.soma_address == setup.v0_address)
        .expect("Should find validator 0");
    assert_eq!(
        validator.next_epoch_commission_rate, new_rate,
        "Commission rate should be set for next epoch"
    );
}

#[tokio::test]
async fn test_set_commission_rate_not_a_validator() {
    let (setup, _v0_key) = setup_validator_test().await;

    let (random_sender, random_key): (_, Ed25519KeyPair) = get_key_pair();
    setup
        .authority_state
        .database_for_testing()
        .set_balance(random_sender, types::object::CoinType::Usdc, 10_000_000)
        .unwrap();

    let data = TransactionData::new(
        TransactionKind::SetCommissionRate { new_rate: 500 },
        random_sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &random_key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(
                !effects.status().is_ok(),
                "Non-validator should not be able to set commission rate"
            );
        }
        Err(_) => {
            // Also acceptable
        }
    }
}

// =============================================================================
// UpdateValidatorMetadata tests
// =============================================================================

#[tokio::test]
async fn test_update_validator_metadata_success() {
    let (setup, v0_key) = setup_validator_test().await;

    // Network address must be BCS-serialized String, not raw bytes
    let addr_str = "/ip4/127.0.0.1/tcp/9999".to_string();
    let addr_bytes = bcs::to_bytes(&addr_str).unwrap();

    let data = TransactionData::new(
        TransactionKind::UpdateValidatorMetadata(UpdateValidatorMetadataArgs {
            next_epoch_network_address: Some(addr_bytes),
            next_epoch_p2p_address: None,
            next_epoch_primary_address: None,
            next_epoch_proxy_address: None,
            next_epoch_protocol_pubkey: None,
            next_epoch_worker_pubkey: None,
            next_epoch_network_pubkey: None,
            next_epoch_proof_of_possession: None,
        }),
        setup.v0_address,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &v0_key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    assert!(result.is_ok(), "UpdateValidatorMetadata should succeed: {:?}", result.err());
    let (_, effects) = result.unwrap();
    assert_eq!(*effects.status(), ExecutionStatus::Success);
}

// =============================================================================
// ReportValidator tests
// =============================================================================

#[tokio::test]
async fn test_report_validator_not_a_validator_reporter() {
    // A non-validator signer reporting should fail with NotAValidator
    let (setup, _v0_key) = setup_validator_test().await;

    let (random_sender, random_key): (_, Ed25519KeyPair) = get_key_pair();
    setup
        .authority_state
        .database_for_testing()
        .set_balance(random_sender, types::object::CoinType::Usdc, 10_000_000)
        .unwrap();

    let data = TransactionData::new(
        TransactionKind::ReportValidator { reportee: setup.v0_address },
        random_sender,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &random_key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Non-validator should not be able to report");
        }
        Err(_) => {
            // Also acceptable if rejected at input validation
        }
    }
}

#[tokio::test]
async fn test_report_validator_nonexistent_reportee() {
    // Reporting a non-validator address should fail with NotAValidator
    let (setup, v0_key) = setup_validator_test().await;

    let (fake_reportee, _): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let data = TransactionData::new(
        TransactionKind::ReportValidator { reportee: fake_reportee },
        setup.v0_address,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &v0_key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(
                !effects.status().is_ok(),
                "Reporting non-validator should fail with NotAValidator"
            );
        }
        Err(_) => {
            // Also acceptable
        }
    }
}

#[tokio::test]
async fn test_report_validator_self_report() {
    // Validator reporting themselves should fail with CannotReportOneself
    let (setup, v0_key) = setup_validator_test().await;

    let data = TransactionData::new(
        TransactionKind::ReportValidator { reportee: setup.v0_address },
        setup.v0_address,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &v0_key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Validator should not be able to report themselves");
        }
        Err(_) => {
            // Also acceptable
        }
    }
}

// =============================================================================
// UndoReportValidator tests
// =============================================================================

#[tokio::test]
async fn test_undo_report_validator_no_record() {
    // Undoing a report that doesn't exist should fail
    let (setup, v0_key) = setup_validator_test().await;

    let (fake_reportee, _): (SomaAddress, Ed25519KeyPair) = get_key_pair();
    let data = TransactionData::new(
        TransactionKind::UndoReportValidator { reportee: fake_reportee },
        setup.v0_address,
        vec![],
    );
    let tx = to_sender_signed_transaction(data, &v0_key);
    let result = send_and_confirm_transaction_(&setup.authority_state, None, tx, true).await;

    match result {
        Ok((_, effects)) => {
            assert!(!effects.status().is_ok(), "Should fail: no report to undo");
        }
        Err(_) => {
            // Also acceptable
        }
    }
}
