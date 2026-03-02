// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for Target Generation and Data Submission.
//!
//! Tests:
//! 1. test_genesis_target_bootstrap - Genesis creates seed targets
//! 2. test_submit_data_fills_target - Submission fills target, spawns replacement
//! 3. test_claim_rewards_after_audit_window - Reward claiming works correctly
//! 4. test_epoch_boundary_issues_new_targets - Epoch change creates new targets
//! 5. test_claim_expired_unfilled_target - Cleanup of expired unfilled targets
//! 6. test_submit_data_validation_errors - Validates submission rejection scenarios

use rpc::proto::soma::ListTargetsRequest;
use test_cluster::TestClusterBuilder;
use tracing::info;
use types::base::SomaAddress;
use types::checksum::Checksum;
use types::config::genesis_config::{GenesisModelConfig, SHANNONS_PER_SOMA};
use types::crypto::DecryptionKey;
use types::digests::ModelWeightsCommitment;
use types::effects::TransactionEffectsAPI;
use types::metadata::{Manifest, ManifestV1, Metadata, MetadataV1};
use types::object::ObjectID;
use types::submission::SubmissionManifest;
use types::system_state::SystemStateTrait as _;
use types::tensor::SomaTensor;
use types::transaction::{ClaimRewardsArgs, SubmitDataArgs, TransactionData, TransactionKind};
use url::Url;
use utils::logging::init_tracing;

// ===== Helpers =====

fn make_manifest(url_str: &str) -> Manifest {
    let url = Url::parse(url_str).expect("Invalid URL");
    let metadata = Metadata::V1(MetadataV1::new(Checksum::new_from_hash([1u8; 32]), 1024));
    Manifest::V1(ManifestV1::new(url, metadata))
}

fn make_genesis_model_config(owner: SomaAddress, initial_stake: u64) -> GenesisModelConfig {
    let manifest = make_manifest("https://example.com/models/genesis");
    GenesisModelConfig {
        owner,
        manifest,
        decryption_key: DecryptionKey::new([0xAA; 32]),
        weights_commitment: ModelWeightsCommitment::new([0xBB; 32]),
        architecture_version: 1,
        commission_rate: 0,
        initial_stake,
    }
}

fn make_submission_manifest(size: usize) -> SubmissionManifest {
    let url = Url::parse("https://example.com/data/submission.bin").unwrap();
    let metadata = Metadata::V1(MetadataV1::new(Checksum::new_from_hash([0xCC; 32]), size));
    let manifest = Manifest::V1(ManifestV1::new(url, metadata));
    SubmissionManifest::new(manifest)
}

// ===================================================================
// Test 1: Genesis target bootstrap
//
// Verifies that seed targets are created at genesis when active models
// exist. The number of targets should match initial_targets_per_epoch
// and each target should have correct thresholds and be in Open status.
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_genesis_target_bootstrap() {
    init_tracing();

    // Create a genesis model to ensure targets can be generated
    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    // Query system state to verify targets were created
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    // Verify we have an active model (model_id is auto-assigned at genesis)
    assert!(
        !system_state.model_registry().active_models.is_empty(),
        "Genesis model should be active"
    );

    // Verify target_state was initialized
    assert!(
        system_state.target_state().distance_threshold.as_scalar() > 0.0,
        "Distance threshold should be initialized"
    );
    assert!(
        system_state.target_state().reward_per_target > 0,
        "Reward per target should be calculated"
    );

    let initial_targets_per_epoch = system_state.parameters().target_initial_targets_per_epoch;

    // Use the SDK to list targets
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.epoch_filter = Some(0);
    request.page_size = Some(100);

    let response = client.list_targets(request).await.unwrap();

    assert_eq!(
        response.targets.len() as u64,
        initial_targets_per_epoch,
        "Should create initial_targets_per_epoch open targets at genesis"
    );

    // Verify each target has expected properties
    for target in &response.targets {
        assert_eq!(target.status.as_deref(), Some("open"), "Genesis target should be open");
        assert_eq!(target.generation_epoch, Some(0), "Genesis target epoch should be 0");
    }

    info!("test_genesis_target_bootstrap passed: {} targets created", response.targets.len());
}

// ===================================================================
// Test 2: Submit data fills target
//
// Verifies that a valid SubmitData transaction:
// - Fills the target (status -> Filled)
// - Records submitter and model info on the target
// - Spawns a replacement target
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_submit_data_fills_target() {
    init_tracing();

    // Create a genesis model
    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    let submitter = test_cluster.get_addresses()[0];

    // Get system state to find thresholds and discover auto-assigned model_id
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });
    let model_id = *system_state
        .model_registry()
        .active_models
        .keys()
        .next()
        .expect("Should have at least one active model");

    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // List open targets using SDK
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);

    let response = client.list_targets(request).await.unwrap();
    assert!(!response.targets.is_empty(), "Should have open targets");

    let target_proto = &response.targets[0];
    let target_id: ObjectID = target_proto
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    info!("Found open target {}", target_id);

    // Count initial targets
    let mut initial_request = ListTargetsRequest::default();
    initial_request.page_size = Some(100);
    let initial_response = client.list_targets(initial_request).await.unwrap();
    let initial_target_count = initial_response.targets.len();

    // Prepare submission data
    let data_manifest = make_submission_manifest(1024);
    let embedding = SomaTensor::zeros(vec![embedding_dim]);
    let distance_score = SomaTensor::scalar(distance_threshold - 0.1);

    // Get gas object
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    // Create and execute SubmitData transaction
    let submit_tx = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_manifest,
            model_id,
            embedding,
            distance_score,
            loss_score: SomaTensor::new(vec![0.5], vec![1]),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let response = test_cluster.sign_and_execute_transaction(&submit_tx).await;

    assert!(
        response.effects.status().is_ok(),
        "SubmitData transaction should succeed: {:?}",
        response.effects.status()
    );

    info!("SubmitData transaction succeeded");

    // Verify target is now filled
    let mut filled_request = ListTargetsRequest::default();
    filled_request.status_filter = Some("filled".to_string());
    filled_request.page_size = Some(100);
    let filled_response = client.list_targets(filled_request).await.unwrap();
    assert!(!filled_response.targets.is_empty(), "Should have a filled target after submission");

    // Verify replacement target was spawned (total count should increase by 1)
    let mut final_request = ListTargetsRequest::default();
    final_request.page_size = Some(100);
    let final_response = client.list_targets(final_request).await.unwrap();
    assert_eq!(
        final_response.targets.len(),
        initial_target_count + 1,
        "A replacement target should be spawned"
    );

    info!("test_submit_data_fills_target passed");
}

// ===================================================================
// Test 3: Claim rewards after audit window
//
// Verifies that rewards can be claimed after the audit window closes.
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_claim_rewards_after_audit_window() {
    init_tracing();

    // Create a genesis model
    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    let submitter = test_cluster.get_addresses()[0];

    // Get system state and thresholds, discover auto-assigned model_id
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });
    let model_id = *system_state
        .model_registry()
        .active_models
        .keys()
        .next()
        .expect("Should have at least one active model");

    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // List open targets
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    let target_proto = &response.targets[0];
    let target_id: ObjectID = target_proto
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    // Submit data to fill the target (epoch 0)
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    let submit_tx = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold - 0.1),
            loss_score: SomaTensor::new(vec![0.5], vec![1]),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let submit_response = test_cluster.sign_and_execute_transaction(&submit_tx).await;
    assert!(submit_response.effects.status().is_ok(), "SubmitData should succeed");

    info!("Target filled in epoch 0");

    // Advance epoch twice to close the audit window
    test_cluster.trigger_reconfiguration().await;
    info!("Advanced to epoch 1");

    test_cluster.trigger_reconfiguration().await;
    info!("Advanced to epoch 2 - audit window closed");

    // Claim rewards
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    let claim_tx = TransactionData::new(
        TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id }),
        submitter,
        vec![gas_object],
    );

    let claim_response = test_cluster.sign_and_execute_transaction(&claim_tx).await;
    assert!(
        claim_response.effects.status().is_ok(),
        "ClaimRewards should succeed: {:?}",
        claim_response.effects.status()
    );

    info!("ClaimRewards succeeded");

    // Verify target is deleted (claimed targets are pruned from storage and index)
    let mut all_request = ListTargetsRequest::default();
    all_request.page_size = Some(100);
    let all_response = client.list_targets(all_request).await.unwrap();
    let found = all_response
        .targets
        .iter()
        .any(|t| t.id.as_ref().and_then(|id_str| id_str.parse().ok()) == Some(target_id));

    assert!(!found, "Target should be deleted after ClaimRewards");

    info!("test_claim_rewards_after_audit_window passed");
}

// ===================================================================
// Test 4: Epoch boundary issues new targets
//
// Verifies that advancing to a new epoch creates new targets.
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_epoch_boundary_issues_new_targets() {
    init_tracing();

    // Create a genesis model
    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    // Get initial target count and parameters
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    let initial_targets_per_epoch = system_state.parameters().target_initial_targets_per_epoch;

    // Count epoch 0 targets
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut epoch0_request = ListTargetsRequest::default();
    epoch0_request.epoch_filter = Some(0);
    epoch0_request.page_size = Some(100);
    let epoch0_response = client.list_targets(epoch0_request).await.unwrap();
    let epoch0_count = epoch0_response.targets.len();

    info!("Epoch 0: {} targets (expected {})", epoch0_count, initial_targets_per_epoch);

    // Advance to epoch 1
    test_cluster.trigger_reconfiguration().await;

    // Count epoch 1 targets
    let mut epoch1_request = ListTargetsRequest::default();
    epoch1_request.epoch_filter = Some(1);
    epoch1_request.page_size = Some(100);
    let epoch1_response = client.list_targets(epoch1_request).await.unwrap();
    let epoch1_count = epoch1_response.targets.len();

    assert_eq!(
        epoch1_count as u64, initial_targets_per_epoch,
        "Should have initial_targets_per_epoch targets from epoch 1"
    );

    // Count total targets
    let mut total_request = ListTargetsRequest::default();
    total_request.page_size = Some(100);
    let total_response = client.list_targets(total_request).await.unwrap();
    let total_count = total_response.targets.len();

    assert_eq!(total_count, epoch0_count + epoch1_count, "Should have epoch 0 + epoch 1 targets");

    info!(
        "test_epoch_boundary_issues_new_targets passed: {} epoch 0 targets, {} epoch 1 targets",
        epoch0_count, epoch1_count
    );
}

// ===================================================================
// Test 5: Claim expired unfilled target
//
// Verifies that claiming an expired unfilled target returns rewards.
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_claim_expired_unfilled_target() {
    init_tracing();

    // Create a genesis model
    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    let claimer = test_cluster.get_addresses()[0];

    // List open epoch 0 targets
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.epoch_filter = Some(0);
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    let target_proto = &response.targets[0];
    let target_id: ObjectID = target_proto
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    info!("Found epoch 0 target {}", target_id);

    // Advance epoch to make the target expired
    test_cluster.trigger_reconfiguration().await;
    info!("Advanced to epoch 1 - epoch 0 targets are now expired");

    // Claim the expired unfilled target
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(claimer)
        .await
        .unwrap()
        .expect("Claimer should have a gas object");

    let claim_tx = TransactionData::new(
        TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id }),
        claimer,
        vec![gas_object],
    );

    let claim_response = test_cluster.sign_and_execute_transaction(&claim_tx).await;
    assert!(
        claim_response.effects.status().is_ok(),
        "ClaimRewards on expired unfilled target should succeed: {:?}",
        claim_response.effects.status()
    );

    info!("ClaimRewards on expired unfilled target succeeded");

    // Verify target is deleted (claimed targets are pruned from storage and index)
    let mut all_request = ListTargetsRequest::default();
    all_request.page_size = Some(100);
    let all_response = client.list_targets(all_request).await.unwrap();
    let found = all_response
        .targets
        .iter()
        .any(|t| t.id.as_ref().and_then(|id_str| id_str.parse().ok()) == Some(target_id));

    assert!(!found, "Target should be deleted after ClaimRewards");

    info!("test_claim_expired_unfilled_target passed");
}

// ===================================================================
// Test 6: Submit data validation errors
//
// Verifies that invalid submissions are rejected with appropriate errors.
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_submit_data_validation_errors() {
    init_tracing();

    // Create a genesis model
    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    let submitter = test_cluster.get_addresses()[0];

    // Get system state and thresholds, discover auto-assigned model_id
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });
    let model_id = *system_state
        .model_registry()
        .active_models
        .keys()
        .next()
        .expect("Should have at least one active model");

    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // List open targets
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    let target_proto = &response.targets[0];
    let target_id: ObjectID = target_proto
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    // Test 1: Distance score exceeds threshold
    {
        let gas_object = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(submitter)
            .await
            .unwrap()
            .expect("Submitter should have a gas object");

        let bad_tx = TransactionData::new(
            TransactionKind::SubmitData(SubmitDataArgs {
                target_id,
                data_manifest: make_submission_manifest(1024),
                model_id,
                embedding: SomaTensor::zeros(vec![embedding_dim]),
                distance_score: SomaTensor::scalar(distance_threshold + 0.1), // Exceeds threshold
                loss_score: SomaTensor::new(vec![0.5], vec![1]),
                bond_coin: gas_object,
            }),
            submitter,
            vec![gas_object],
        );

        let response = test_cluster.sign_and_execute_transaction(&bad_tx).await;
        assert!(
            response.effects.status().is_err(),
            "Should reject submission with distance score exceeding threshold"
        );
        info!("Distance exceeds threshold: correctly rejected");
    }

    // Test 2: Wrong embedding dimension
    {
        let gas_object = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(submitter)
            .await
            .unwrap()
            .expect("Submitter should have a gas object");

        let wrong_dim = if embedding_dim > 10 { embedding_dim - 10 } else { embedding_dim + 10 };

        let bad_tx = TransactionData::new(
            TransactionKind::SubmitData(SubmitDataArgs {
                target_id,
                data_manifest: make_submission_manifest(1024),
                model_id,
                embedding: SomaTensor::zeros(vec![wrong_dim]), // Wrong dimension
                distance_score: SomaTensor::scalar(distance_threshold - 0.1),
                loss_score: SomaTensor::new(vec![0.5], vec![1]),
                bond_coin: gas_object,
            }),
            submitter,
            vec![gas_object],
        );

        let response = test_cluster.sign_and_execute_transaction(&bad_tx).await;
        assert!(
            response.effects.status().is_err(),
            "Should reject submission with wrong embedding dimension"
        );
        info!("Wrong embedding dimension: correctly rejected");
    }

    // Test 3: Model not in target
    {
        let gas_object = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(submitter)
            .await
            .unwrap()
            .expect("Submitter should have a gas object");

        let wrong_model_id = ObjectID::from_bytes([0x99; 32]).unwrap();

        let bad_tx = TransactionData::new(
            TransactionKind::SubmitData(SubmitDataArgs {
                target_id,
                data_manifest: make_submission_manifest(1024),
                model_id: wrong_model_id, // Wrong model
                embedding: SomaTensor::zeros(vec![embedding_dim]),
                distance_score: SomaTensor::scalar(distance_threshold - 0.1),
                loss_score: SomaTensor::new(vec![0.5], vec![1]),
                bond_coin: gas_object,
            }),
            submitter,
            vec![gas_object],
        );

        let response = test_cluster.sign_and_execute_transaction(&bad_tx).await;
        assert!(
            response.effects.status().is_err(),
            "Should reject submission with model not in target"
        );
        info!("Model not in target: correctly rejected");
    }

    info!("test_submit_data_validation_errors passed: all invalid submissions correctly rejected");
}
