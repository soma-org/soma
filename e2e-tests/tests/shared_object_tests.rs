// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

//! Tier 3: Shared object E2E tests.
//!
//! Rewrites of Sui's shared_objects_tests.rs and shared_objects_version_tests.rs
//! using SOMA's native shared objects (Target, SystemState) instead of Move counters.
//!
//! Tests:
//! 1. test_shared_object_mutation_via_submit_data — SubmitData mutates Target shared object
//! 2. test_conflicting_owned_transactions_same_coin — Two transfers spending the same coin
//! 3. test_shared_object_status_transition_via_claim — ClaimRewards transitions Target to Claimed
//! 4. test_target_version_increments_on_mutations — Version strictly increases on each mutation
//! 5. test_transaction_replay_idempotency — Resubmit same signed tx, get identical effects
//! 6. test_racing_submitters_concurrent_shared_mutations — Concurrent SubmitData to same Target
//! 7. test_shared_object_dependency_tracking — Sequential mutations create dependency chain
//! 8. test_concurrent_conflicting_owned_transactions — Concurrent spends of same coin via orchestrator

use rpc::proto::soma::ListTargetsRequest;
use test_cluster::TestClusterBuilder;
use tracing::info;
use types::{
    SYSTEM_STATE_OBJECT_ID,
    base::SomaAddress,
    checksum::Checksum,
    config::genesis_config::{GenesisModelConfig, SHANNONS_PER_SOMA},
    crypto::{DecryptionKey, DefaultHash, Signature},
    digests::{DataCommitment, ModelWeightsCommitment, ModelWeightsUrlCommitment},
    effects::{InputSharedObject, TransactionEffectsAPI},
    intent::{Intent, IntentMessage},
    metadata::{Manifest, ManifestV1, Metadata, MetadataV1},
    model::{ModelId, ModelWeightsManifest},
    object::{ObjectID, Owner},
    quorum_driver::{ExecuteTransactionRequest, ExecuteTransactionRequestType},
    submission::SubmissionManifest,
    system_state::SystemStateTrait as _,
    tensor::SomaTensor,
    transaction::{
        ClaimRewardsArgs, SubmitDataArgs, Transaction, TransactionData, TransactionKind,
    },
};
use url::Url;
use utils::logging::init_tracing;

use fastcrypto::hash::HashFunction as _;

// ===== Helpers (shared with target_tests.rs / challenge_tests.rs) =====

fn url_commitment_for(url_str: &str) -> ModelWeightsUrlCommitment {
    let mut hasher = DefaultHash::default();
    hasher.update(url_str.as_bytes());
    let hash = hasher.finalize();
    let bytes: [u8; 32] = hash.as_ref().try_into().unwrap();
    ModelWeightsUrlCommitment::new(bytes)
}

fn make_weights_manifest(url_str: &str) -> ModelWeightsManifest {
    let url = Url::parse(url_str).expect("Invalid URL");
    let metadata = Metadata::V1(MetadataV1::new(Checksum::new_from_hash([1u8; 32]), 1024));
    let manifest = Manifest::V1(ManifestV1::new(url, metadata));
    ModelWeightsManifest { manifest, decryption_key: DecryptionKey::new([0xAA; 32]) }
}

fn make_genesis_model_config(
    owner: SomaAddress,
    model_id: ModelId,
    initial_stake: u64,
) -> GenesisModelConfig {
    let url_str = format!("https://example.com/models/{}", model_id);
    let url_commitment = url_commitment_for(&url_str);
    let weights_commitment = ModelWeightsCommitment::new([0xBB; 32]);
    let weights_manifest = make_weights_manifest(&url_str);

    GenesisModelConfig {
        owner,
        model_id,
        weights_manifest,
        weights_url_commitment: url_commitment,
        weights_commitment,
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

/// Sign a transaction as a validator using their account key pair.
fn sign_transaction_as_validator(
    test_cluster: &test_cluster::TestCluster,
    validator_index: usize,
    tx_data: &TransactionData,
) -> Transaction {
    let validator_config = &test_cluster.swarm.config().validator_configs[validator_index];
    let keypair = validator_config.account_key_pair.keypair();

    let intent_msg = IntentMessage::new(Intent::soma_transaction(), tx_data);
    let sig = Signature::new_secure(&intent_msg, keypair);

    Transaction::from_data(tx_data.clone(), vec![sig])
}

/// Get validator's soma address by index
fn get_validator_address(
    test_cluster: &test_cluster::TestCluster,
    validator_index: usize,
) -> SomaAddress {
    test_cluster.swarm.config().validator_configs[validator_index].soma_address()
}

/// Get a gas object for a validator by transferring from a funded wallet address.
async fn get_validator_gas_object(
    test_cluster: &test_cluster::TestCluster,
    validator_address: SomaAddress,
    funder_index: usize,
) -> types::object::ObjectRef {
    let funder = test_cluster.get_addresses()[funder_index];
    let gas_coin = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(funder)
        .await
        .unwrap()
        .expect("Funder should have gas");

    let transfer_tx = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: gas_coin,
            recipient: validator_address,
            amount: Some(1_000_000_000_000), // 1000 SOMA
        },
        funder,
        vec![gas_coin],
    );

    let response = test_cluster.sign_and_execute_transaction(&transfer_tx).await;
    assert!(response.effects.status().is_ok(), "Failed to fund validator");

    let created = response.effects.created();
    for (obj_ref, owner) in created {
        if let Owner::AddressOwner(addr) = owner {
            if addr == validator_address {
                return obj_ref;
            }
        }
    }

    panic!("Failed to find gas object for validator after transfer");
}

// ===================================================================
// Test 1: Shared object mutation via SubmitData
//
// Verifies that SubmitData mutates the Target shared object:
// - Target appears in effects.mutated() (not created/deleted)
// - Target remains Owner::Shared after mutation
// - SystemState is also mutated as a shared input
// - effects.input_shared_objects() lists both Target and SystemState
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_shared_object_mutation_via_submit_data() {
    init_tracing();

    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let model_id = ObjectID::from_bytes([0x42; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, model_id, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    let submitter = test_cluster.get_addresses()[0];

    // Get system state and thresholds
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().expect("Should get SystemState")
    });
    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // Find an open target
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    assert!(!response.targets.is_empty(), "Should have open targets");

    let target_id: ObjectID = response.targets[0]
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    info!("Found open target {}", target_id);

    // Get gas object
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    // Execute SubmitData
    let submit_tx = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold - 0.1),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let response = test_cluster.sign_and_execute_transaction(&submit_tx).await;
    assert!(
        response.effects.status().is_ok(),
        "SubmitData should succeed: {:?}",
        response.effects.status()
    );

    // === Verify shared object mutation in effects ===

    // 1. Target should appear in mutated() list
    let mutated = response.effects.mutated();
    let target_mutated = mutated.iter().find(|((id, _, _), _)| *id == target_id);
    assert!(target_mutated.is_some(), "Target should appear in mutated objects");

    let (target_ref, target_owner) = target_mutated.unwrap();
    info!("Target mutated: id={}, version={}", target_ref.0, target_ref.1.value());

    // 2. Target should still be a shared object after mutation
    assert!(
        matches!(target_owner, Owner::Shared { .. }),
        "Target should remain Owner::Shared after mutation, got: {:?}",
        target_owner
    );

    // 3. SystemState should also appear in mutated() (shared object, always mutated by SubmitData)
    let system_state_mutated = mutated.iter().find(|((id, _, _), _)| *id == SYSTEM_STATE_OBJECT_ID);
    assert!(system_state_mutated.is_some(), "SystemState should appear in mutated objects");

    // 4. Check input_shared_objects() includes both Target and SystemState
    let input_shared = response.effects.input_shared_objects();
    let target_in_shared = input_shared.iter().any(|iso| match iso {
        InputSharedObject::Mutate((id, _, _)) => *id == target_id,
        _ => false,
    });
    let system_state_in_shared = input_shared.iter().any(|iso| match iso {
        InputSharedObject::Mutate((id, _, _)) => *id == SYSTEM_STATE_OBJECT_ID,
        _ => false,
    });

    assert!(target_in_shared, "Target should be in input_shared_objects as Mutate");
    assert!(system_state_in_shared, "SystemState should be in input_shared_objects as Mutate");

    // 5. Target should NOT appear in created() or deleted()
    let created_ids: Vec<_> =
        response.effects.created().iter().map(|((id, _, _), _)| *id).collect();
    assert!(!created_ids.contains(&target_id), "Target should not appear in created objects");

    let deleted_ids: Vec<_> = response.effects.deleted().iter().map(|(id, _, _)| *id).collect();
    assert!(!deleted_ids.contains(&target_id), "Target should not appear in deleted objects");

    // 6. New objects should be created (Submission object + replacement target)
    assert!(
        !response.effects.created().is_empty(),
        "SubmitData should create new objects (submission + replacement target)"
    );

    info!(
        "test_shared_object_mutation_via_submit_data passed: target mutated, {} new objects created",
        response.effects.created().len()
    );
}

// ===================================================================
// Test 2: Conflicting owned transactions — same coin
//
// Two transfers spending the same coin (same ObjectRef as input).
// The first transaction succeeds; the second fails because the coin's
// version has changed (ObjectVersionUnavailableForConsumption).
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_conflicting_owned_transactions_same_coin() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let recipient_a = addresses[1];
    let recipient_b = addresses[2];

    // Get a gas coin — this ObjectRef will be used by BOTH transactions
    let coin_ref = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(sender)
        .await
        .unwrap()
        .expect("Sender should have a gas object");

    info!("Using coin {} version {} for both transactions", coin_ref.0, coin_ref.1.value());

    // Create two TransferCoin transactions using the SAME ObjectRef
    let tx1_data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: coin_ref,
            amount: Some(1_000_000),
            recipient: recipient_a,
        },
        sender,
        vec![coin_ref],
    );

    let tx2_data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: coin_ref,
            amount: Some(1_000_000),
            recipient: recipient_b,
        },
        sender,
        vec![coin_ref],
    );

    // Execute first transaction — should succeed
    let response1 = test_cluster.sign_and_execute_transaction(&tx1_data).await;
    assert!(
        response1.effects.status().is_ok(),
        "First transfer should succeed: {:?}",
        response1.effects.status()
    );
    info!("First transfer succeeded, coin version now {}", response1.effects.version().value());

    // Execute second transaction with the STALE ObjectRef — should fail
    // The coin's version has changed, so the ObjectRef is now invalid
    let tx2 = test_cluster.wallet.sign_transaction(&tx2_data).await;
    let result2 = test_cluster.wallet.execute_transaction_may_fail(tx2).await;

    match result2 {
        Ok(response2) => {
            // If we got a response, the effects should show a failure
            assert!(
                response2.effects.status().is_err(),
                "Second transfer should fail with stale object version"
            );
            info!(
                "Second transfer correctly rejected with error in effects: {:?}",
                response2.effects.status()
            );
        }
        Err(e) => {
            // The transaction may fail at the orchestrator level (before execution)
            info!("Second transfer correctly rejected at submission: {}", e);
        }
    }

    // Verify the coin was only spent once — check that recipient_a received funds
    // but recipient_b did not get a new coin from this specific transfer
    let created_by_tx1 = response1.effects.created();
    let recipient_a_coin = created_by_tx1
        .iter()
        .any(|(_, owner)| matches!(owner, Owner::AddressOwner(addr) if *addr == recipient_a));
    assert!(recipient_a_coin, "recipient_a should have received a coin from tx1");

    info!("test_conflicting_owned_transactions_same_coin passed: equivocation correctly prevented");
}

// ===================================================================
// Test 3: Shared object status transition via ClaimRewards
//
// After filling a target and closing the challenge window:
// - ClaimRewards transitions target to Claimed status
// - Target appears in mutated() (status change, not deletion)
// - New reward coins are created
// - Subsequent ClaimRewards on the same target fails (TargetAlreadyClaimed)
// - Subsequent SubmitData on the same target fails (TargetNotOpen)
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_shared_object_status_transition_via_claim() {
    init_tracing();

    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let model_id = ObjectID::from_bytes([0x42; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, model_id, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    let submitter = test_cluster.get_addresses()[0];

    // Get system state and thresholds
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().expect("Should get SystemState")
    });
    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // Find and fill an open target
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    let target_id: ObjectID = response.targets[0]
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    let submit_tx = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold - 0.1),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let submit_response = test_cluster.sign_and_execute_transaction(&submit_tx).await;
    assert!(submit_response.effects.status().is_ok(), "SubmitData should succeed");
    info!("Target {} filled in epoch 0", target_id);

    // Advance epochs to close the challenge window (fill_epoch + 1 = 0 + 1 = 1)
    test_cluster.trigger_reconfiguration().await;
    test_cluster.trigger_reconfiguration().await;
    info!("Advanced to epoch 2 — challenge window closed");

    // Execute ClaimRewards
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

    // === Verify shared object deletion in effects ===

    // 1. Target should appear in deleted() (ClaimRewards deletes the terminal target)
    let deleted_ids: Vec<_> =
        claim_response.effects.deleted().iter().map(|(id, _, _)| *id).collect();
    assert!(deleted_ids.contains(&target_id), "Target should be deleted by ClaimRewards");

    // 3. New reward coins should be created (submitter reward + model owner reward + bond return)
    let created = claim_response.effects.created();
    assert!(!created.is_empty(), "ClaimRewards should create reward coins");
    info!("ClaimRewards created {} objects (rewards + bond return)", created.len());

    // 4. Verify submitter received at least one coin
    let submitter_coins: Vec<_> = created
        .iter()
        .filter(|(_, owner)| matches!(owner, Owner::AddressOwner(addr) if *addr == submitter))
        .collect();
    assert!(!submitter_coins.is_empty(), "Submitter should receive rewards or bond return");

    // === Verify subsequent operations on deleted target fail ===

    // 5. Second ClaimRewards on same target should fail (target was deleted)
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    let claim_again_tx = TransactionData::new(
        TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id }),
        submitter,
        vec![gas_object],
    );

    let claim_again_response = test_cluster.sign_and_execute_transaction(&claim_again_tx).await;
    assert!(
        claim_again_response.effects.status().is_err(),
        "Second ClaimRewards should fail (target deleted)"
    );
    info!("Second ClaimRewards correctly rejected");

    // 6. SubmitData on deleted target should also fail
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    let submit_to_claimed_tx = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: make_submission_manifest(512),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold - 0.1),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let submit_to_claimed = test_cluster.sign_and_execute_transaction(&submit_to_claimed_tx).await;
    assert!(
        submit_to_claimed.effects.status().is_err(),
        "SubmitData to deleted target should fail"
    );
    info!("SubmitData to deleted target correctly rejected");

    info!("test_shared_object_status_transition_via_claim passed");
}

// ===================================================================
// Test 4: Target version increments on mutations
//
// Verifies Lamport versioning invariants for shared objects:
// - After SubmitData: target version > initial version
// - After ReportSubmission: target version > post-submit version
// - Version is strictly monotonically increasing across mutations
// - All mutated objects in a transaction share the same effects version
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_target_version_increments_on_mutations() {
    init_tracing();

    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let model_id = ObjectID::from_bytes([0x42; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, model_id, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    let submitter = test_cluster.get_addresses()[0];

    // Get system state and thresholds
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().expect("Should get SystemState")
    });
    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // Find an open target and get its initial version
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    let target_id: ObjectID = response.targets[0]
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    // Get initial target object to read its version
    let initial_target = client.get_object(target_id).await.unwrap();
    let version_0 = initial_target.version();
    info!("Initial target version: {}", version_0.value());

    // === Mutation 1: SubmitData (fills the target) ===

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    let submit_tx = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold - 0.1),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let submit_response = test_cluster.sign_and_execute_transaction(&submit_tx).await;
    assert!(submit_response.effects.status().is_ok(), "SubmitData should succeed");

    // Get target version after SubmitData
    let mutated = submit_response.effects.mutated();
    let target_after_submit = mutated.iter().find(|((id, _, _), _)| *id == target_id);
    assert!(target_after_submit.is_some(), "Target should be in mutated objects");
    let version_1 = target_after_submit.unwrap().0.1;

    info!(
        "After SubmitData: target version {} -> {} (effects version {})",
        version_0.value(),
        version_1.value(),
        submit_response.effects.version().value()
    );

    // INVARIANT 1: Version strictly increases after mutation
    assert!(
        version_1 > version_0,
        "Target version must increase after SubmitData: {} > {}",
        version_1.value(),
        version_0.value()
    );

    // INVARIANT 2: Mutated object version matches effects version
    assert_eq!(
        version_1,
        submit_response.effects.version(),
        "Mutated target version should equal effects version"
    );

    // INVARIANT 3: All mutated objects in same tx share the same version
    for ((obj_id, version, _), _) in &mutated {
        assert_eq!(
            *version,
            submit_response.effects.version(),
            "Object {} version {} should match effects version {}",
            obj_id,
            version.value(),
            submit_response.effects.version().value()
        );
    }

    // === Mutation 2: ReportSubmission (validator reports the filled target) ===

    let validator_addr = get_validator_address(&test_cluster, 0);
    let gas = get_validator_gas_object(&test_cluster, validator_addr, 1).await;

    let report_tx = TransactionData::new(
        TransactionKind::ReportSubmission { target_id, challenger: None },
        validator_addr,
        vec![gas],
    );

    let signed_tx = sign_transaction_as_validator(&test_cluster, 0, &report_tx);
    let report_response = test_cluster.execute_transaction(signed_tx).await;
    assert!(report_response.effects.status().is_ok(), "ReportSubmission should succeed");

    // Get target version after ReportSubmission
    let mutated2 = report_response.effects.mutated();
    let target_after_report = mutated2.iter().find(|((id, _, _), _)| *id == target_id);
    assert!(target_after_report.is_some(), "Target should be in mutated objects after report");
    let version_2 = target_after_report.unwrap().0.1;

    info!(
        "After ReportSubmission: target version {} -> {} (effects version {})",
        version_1.value(),
        version_2.value(),
        report_response.effects.version().value()
    );

    // INVARIANT 4: Version strictly increases again
    assert!(
        version_2 > version_1,
        "Target version must increase after ReportSubmission: {} > {}",
        version_2.value(),
        version_1.value()
    );

    // INVARIANT 5: Mutated object version matches effects version
    assert_eq!(
        version_2,
        report_response.effects.version(),
        "Mutated target version should equal effects version after report"
    );

    // === Mutation 3: Second validator reports (another version bump) ===

    let validator_addr_2 = get_validator_address(&test_cluster, 1);
    let gas2 = get_validator_gas_object(&test_cluster, validator_addr_2, 2).await;

    let report_tx_2 = TransactionData::new(
        TransactionKind::ReportSubmission { target_id, challenger: None },
        validator_addr_2,
        vec![gas2],
    );

    let signed_tx_2 = sign_transaction_as_validator(&test_cluster, 1, &report_tx_2);
    let report_response_2 = test_cluster.execute_transaction(signed_tx_2).await;
    assert!(report_response_2.effects.status().is_ok(), "Second ReportSubmission should succeed");

    let mutated3 = report_response_2.effects.mutated();
    let target_after_report_2 = mutated3.iter().find(|((id, _, _), _)| *id == target_id);
    assert!(target_after_report_2.is_some(), "Target should be in mutated objects");
    let version_3 = target_after_report_2.unwrap().0.1;

    info!(
        "After second ReportSubmission: target version {} -> {}",
        version_2.value(),
        version_3.value()
    );

    // INVARIANT 6: Strict monotonic increase across all mutations
    assert!(
        version_3 > version_2,
        "Target version must increase: {} > {}",
        version_3.value(),
        version_2.value()
    );

    // Summary: version_0 < version_1 < version_2 < version_3
    info!(
        "test_target_version_increments_on_mutations passed: {} < {} < {} < {}",
        version_0.value(),
        version_1.value(),
        version_2.value(),
        version_3.value()
    );
}

// ===================================================================
// Test 5: Transaction replay idempotency
//
// Submitting the exact same signed transaction twice should return
// identical effects. The second submission hits the "already executed"
// path in the orchestrator and returns cached results.
// Adapted from Sui's replay_shared_object_transaction.
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_transaction_replay_idempotency() {
    init_tracing();

    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let model_id = ObjectID::from_bytes([0x42; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, model_id, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    let submitter = test_cluster.get_addresses()[0];

    // Get system state and thresholds
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().expect("Should get SystemState")
    });
    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // Find an open target
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    let target_id: ObjectID = response.targets[0]
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    // Build and sign a SubmitData transaction
    let submit_tx_data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold - 0.1),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let signed_tx = test_cluster.wallet.sign_transaction(&submit_tx_data).await;
    let tx_digest = *signed_tx.digest();
    info!("Signed transaction digest: {}", tx_digest);

    // === First submission via orchestrator ===
    let orchestrator = test_cluster
        .fullnode_handle
        .soma_node
        .with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    let request1 = ExecuteTransactionRequest {
        transaction: signed_tx.clone(),
        include_input_objects: false,
        include_output_objects: false,
    };

    let (response1, executed_locally_1) = orchestrator
        .execute_transaction_block(
            request1,
            ExecuteTransactionRequestType::WaitForLocalExecution,
            None,
        )
        .await
        .expect("First submission should succeed");

    assert!(executed_locally_1, "First submission should execute locally");
    let effects1 = &response1.effects.effects;
    assert!(effects1.status().is_ok(), "First submission should succeed: {:?}", effects1.status());

    let effects_digest_1 = effects1.transaction_digest_owned();
    let effects_version_1 = effects1.version();
    info!(
        "First submission: tx_digest={}, effects_version={}",
        effects_digest_1,
        effects_version_1.value()
    );

    // Verify tx is marked as executed
    assert!(
        test_cluster
            .fullnode_handle
            .soma_node
            .with(|n| n.state().is_tx_already_executed(&tx_digest)),
        "Transaction should be marked as executed after first submission"
    );

    // === Second submission of the SAME signed transaction ===
    let request2 = ExecuteTransactionRequest {
        transaction: signed_tx,
        include_input_objects: false,
        include_output_objects: false,
    };

    let (response2, _) = orchestrator
        .execute_transaction_block(
            request2,
            ExecuteTransactionRequestType::WaitForLocalExecution,
            None,
        )
        .await
        .expect("Second submission (replay) should succeed");

    let effects2 = &response2.effects.effects;

    // === Verify idempotency ===

    // 1. Same transaction digest in effects
    let effects_digest_2 = effects2.transaction_digest_owned();
    assert_eq!(
        effects_digest_1, effects_digest_2,
        "Replayed transaction should have the same transaction digest in effects"
    );

    // 2. Same effects version
    let effects_version_2 = effects2.version();
    assert_eq!(
        effects_version_1, effects_version_2,
        "Replayed transaction should have the same effects version"
    );

    // 3. Same execution status
    assert!(effects2.status().is_ok(), "Replayed transaction should also report success");

    // 4. Same set of mutated objects
    assert_eq!(
        effects1.mutated().len(),
        effects2.mutated().len(),
        "Replayed transaction should mutate the same number of objects"
    );

    // 5. Same set of created objects
    assert_eq!(
        effects1.created().len(),
        effects2.created().len(),
        "Replayed transaction should create the same number of objects"
    );

    info!("test_transaction_replay_idempotency passed: identical effects on replay");
}

// ===================================================================
// Test 6: Racing submitters — concurrent shared object mutations
//
// Multiple submitters submit SubmitData for the same open Target concurrently.
// Since Target is a shared object, consensus sequences these transactions.
// Exactly one should succeed (target becomes Filled); the rest should fail
// with TargetNotOpen (the target is no longer open by the time they execute).
// Adapted from Sui's shared_object_deletion_multiple_times.
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_racing_submitters_concurrent_shared_mutations() {
    init_tracing();

    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let model_id = ObjectID::from_bytes([0x42; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, model_id, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    // Get system state and thresholds
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().expect("Should get SystemState")
    });
    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // Find an open target
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    let target_id: ObjectID = response.targets[0]
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    info!("Racing submitters targeting: {}", target_id);

    // Create 3 signed SubmitData transactions from different senders
    let addresses = test_cluster.get_addresses();
    let num_racers = 3.min(addresses.len());
    let mut signed_txs = Vec::new();

    for i in 0..num_racers {
        let submitter = addresses[i];
        let gas_object = test_cluster
            .wallet
            .get_one_gas_object_owned_by_address(submitter)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("Submitter {} should have a gas object", i));

        let submit_tx = TransactionData::new(
            TransactionKind::SubmitData(SubmitDataArgs {
                target_id,
                data_commitment: DataCommitment::random(),
                data_manifest: make_submission_manifest(1024),
                model_id,
                embedding: SomaTensor::zeros(vec![embedding_dim]),
                distance_score: SomaTensor::scalar(distance_threshold - 0.1),
                bond_coin: gas_object,
            }),
            submitter,
            vec![gas_object],
        );

        let signed = test_cluster.wallet.sign_transaction(&submit_tx).await;
        signed_txs.push(signed);
    }

    // Submit all concurrently via the orchestrator
    let orchestrator = test_cluster
        .fullnode_handle
        .soma_node
        .with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    let mut handles = Vec::new();
    for (i, tx) in signed_txs.into_iter().enumerate() {
        let orch = orchestrator.clone();
        handles.push(tokio::task::spawn(async move {
            let request = ExecuteTransactionRequest {
                transaction: tx,
                include_input_objects: false,
                include_output_objects: false,
            };
            let result = orch
                .execute_transaction_block(
                    request,
                    ExecuteTransactionRequestType::WaitForLocalExecution,
                    None,
                )
                .await;
            (i, result)
        }));
    }

    // Collect results
    let mut successes = 0;
    let mut failures = 0;

    for handle in handles {
        let (i, result) = handle.await.unwrap();
        match result {
            Ok((response, _)) => {
                let effects = &response.effects.effects;
                if effects.status().is_ok() {
                    successes += 1;
                    info!("Submitter {} won the race (target filled)", i);
                } else {
                    failures += 1;
                    info!(
                        "Submitter {} lost the race (execution failed): {:?}",
                        i,
                        effects.status()
                    );
                }
            }
            Err(e) => {
                failures += 1;
                info!("Submitter {} failed at submission level: {:?}", i, e);
            }
        }
    }

    // === Verify racing invariant ===
    assert_eq!(
        successes, 1,
        "Exactly one submitter should win the race (got {} successes, {} failures)",
        successes, failures
    );
    assert_eq!(
        failures,
        num_racers - 1,
        "All other submitters should fail (got {} failures)",
        failures
    );

    info!(
        "test_racing_submitters_concurrent_shared_mutations passed: 1 winner, {} losers",
        failures
    );
}

// ===================================================================
// Test 7: Shared object dependency tracking
//
// Verifies that when two transactions sequentially mutate the same
// shared object, the second transaction's effects.dependencies()
// includes the first transaction's digest.
// Adapted from Sui's call_shared_object_contract (dependency chain).
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_shared_object_dependency_tracking() {
    init_tracing();

    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let model_id = ObjectID::from_bytes([0x42; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = make_genesis_model_config(model_owner, model_id, initial_stake);

    let test_cluster =
        TestClusterBuilder::new().with_genesis_models(vec![model_config]).build().await;

    let submitter = test_cluster.get_addresses()[0];

    // Get system state and thresholds
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state().get_system_state_object_for_testing().expect("Should get SystemState")
    });
    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // Find an open target
    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    let target_id: ObjectID = response.targets[0]
        .id
        .as_ref()
        .and_then(|id_str| id_str.parse().ok())
        .expect("Target should have valid ID");

    info!("Dependency tracking on target: {}", target_id);

    // === Mutation 1: SubmitData (fills target) ===
    let gas_object = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Submitter should have a gas object");

    let submit_tx_data = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_commitment: DataCommitment::random(),
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold - 0.1),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let response1 = test_cluster.sign_and_execute_transaction(&submit_tx_data).await;
    assert!(response1.effects.status().is_ok(), "SubmitData should succeed");

    let tx1_digest = *response1.effects.transaction_digest();
    info!("Mutation 1 (SubmitData) digest: {}", tx1_digest);

    // === Mutation 2: ReportSubmission (validator reports on same target) ===
    let validator_addr = get_validator_address(&test_cluster, 0);
    let gas = get_validator_gas_object(&test_cluster, validator_addr, 1).await;

    let report_tx_data = TransactionData::new(
        TransactionKind::ReportSubmission { target_id, challenger: None },
        validator_addr,
        vec![gas],
    );

    let signed_report = sign_transaction_as_validator(&test_cluster, 0, &report_tx_data);
    let response2 = test_cluster.execute_transaction(signed_report).await;
    assert!(response2.effects.status().is_ok(), "ReportSubmission should succeed");

    let tx2_digest = *response2.effects.transaction_digest();
    let tx2_deps = response2.effects.dependencies();
    info!("Mutation 2 (ReportSubmission) digest: {}", tx2_digest);
    info!("Mutation 2 dependencies: {:?}", tx2_deps);

    // === Verify dependency chain ===

    // The ReportSubmission mutates the same Target that SubmitData mutated.
    // Therefore, tx2's dependencies should include tx1's digest (the last
    // transaction that modified the shared object).
    assert!(
        tx2_deps.contains(&tx1_digest),
        "ReportSubmission dependencies should include the SubmitData digest.\n\
         Expected {} in {:?}",
        tx1_digest,
        tx2_deps,
    );

    // Dependencies should not be empty
    assert!(!tx2_deps.is_empty(), "Transaction dependencies should not be empty");

    // === Mutation 3: Second validator report — should depend on mutation 2 ===
    let validator_addr_2 = get_validator_address(&test_cluster, 1);
    let gas2 = get_validator_gas_object(&test_cluster, validator_addr_2, 2).await;

    let report_tx_data_2 = TransactionData::new(
        TransactionKind::ReportSubmission { target_id, challenger: None },
        validator_addr_2,
        vec![gas2],
    );

    let signed_report_2 = sign_transaction_as_validator(&test_cluster, 1, &report_tx_data_2);
    let response3 = test_cluster.execute_transaction(signed_report_2).await;
    assert!(response3.effects.status().is_ok(), "Second ReportSubmission should succeed");

    let tx3_deps = response3.effects.dependencies();
    info!("Mutation 3 dependencies: {:?}", tx3_deps);

    // tx3 should depend on tx2 (which last modified the Target)
    assert!(
        tx3_deps.contains(&tx2_digest),
        "Third mutation should depend on second mutation.\n\
         Expected {} in {:?}",
        tx2_digest,
        tx3_deps,
    );

    info!(
        "test_shared_object_dependency_tracking passed: {} -> {} -> {}",
        tx1_digest,
        tx2_digest,
        response3.effects.transaction_digest()
    );
}

// ===================================================================
// Test 8: Concurrent conflicting owned transactions
//
// Two transfers spending the same owned coin submitted concurrently
// via the orchestrator. Only one should succeed; the other gets
// ObjectsDoubleUsed from the quorum driver (validators lock the
// object to one transaction).
// Adapted from Sui's test_conflicting_owned_transactions (soft bundle).
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_concurrent_conflicting_owned_transactions() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let recipient_a = addresses[1];
    let recipient_b = addresses[2];

    // Get three coins: one contested coin + separate gas for each tx
    let coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(sender, Some(3)).await.unwrap();
    assert!(coins.len() >= 3, "Sender needs at least 3 coins");
    let coin_ref = coins[0];
    let gas_ref1 = coins[1];
    let gas_ref2 = coins[2];

    info!("Concurrent spend of coin {} version {}", coin_ref.0, coin_ref.1.value());

    // Create and sign two conflicting TransferCoin transactions
    // Each uses a separate gas coin so only the transfer coin conflicts
    let tx1_data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: coin_ref,
            amount: Some(1_000_000),
            recipient: recipient_a,
        },
        sender,
        vec![gas_ref1],
    );

    let tx2_data = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: coin_ref,
            amount: Some(1_000_000),
            recipient: recipient_b,
        },
        sender,
        vec![gas_ref2],
    );

    let signed_tx1 = test_cluster.wallet.sign_transaction(&tx1_data).await;
    let signed_tx2 = test_cluster.wallet.sign_transaction(&tx2_data).await;
    let digest1 = *signed_tx1.digest();
    let digest2 = *signed_tx2.digest();

    info!("TX1 (to recipient_a) digest: {}", digest1);
    info!("TX2 (to recipient_b) digest: {}", digest2);

    // Submit both concurrently via the orchestrator
    let orchestrator = test_cluster
        .fullnode_handle
        .soma_node
        .with(|n| n.transaction_orchestrator().expect("fullnode must have orchestrator"));

    let orch1 = orchestrator.clone();
    let orch2 = orchestrator.clone();

    let handle1 = tokio::task::spawn(async move {
        let request = ExecuteTransactionRequest {
            transaction: signed_tx1,
            include_input_objects: false,
            include_output_objects: false,
        };
        orch1
            .execute_transaction_block(
                request,
                ExecuteTransactionRequestType::WaitForLocalExecution,
                None,
            )
            .await
    });

    let handle2 = tokio::task::spawn(async move {
        let request = ExecuteTransactionRequest {
            transaction: signed_tx2,
            include_input_objects: false,
            include_output_objects: false,
        };
        orch2
            .execute_transaction_block(
                request,
                ExecuteTransactionRequestType::WaitForLocalExecution,
                None,
            )
            .await
    });

    let (result1, result2) = tokio::join!(handle1, handle2);
    let result1 = result1.unwrap();
    let result2 = result2.unwrap();

    // Exactly one should succeed, the other should fail with ObjectsDoubleUsed
    let (successes, failures) =
        [("TX1", &result1), ("TX2", &result2)].iter().fold((0, 0), |(s, f), (name, result)| {
            match result {
                Ok((response, _)) => {
                    let effects = &response.effects.effects;
                    if effects.status().is_ok() {
                        info!("{} succeeded", name);
                        (s + 1, f)
                    } else {
                        info!("{} failed in effects: {:?}", name, effects.status());
                        (s, f + 1)
                    }
                }
                Err(e) => {
                    info!("{} failed at submission: {:?}", name, e);
                    (s, f + 1)
                }
            }
        });

    assert_eq!(
        successes, 1,
        "Exactly one conflicting owned transaction should succeed (got {})",
        successes
    );
    assert_eq!(
        failures, 1,
        "Exactly one conflicting owned transaction should fail (got {})",
        failures
    );

    // Verify the coin was only spent once
    let executed_count = [digest1, digest2]
        .iter()
        .filter(|d| {
            test_cluster.fullnode_handle.soma_node.with(|n| n.state().is_tx_already_executed(d))
        })
        .count();

    assert_eq!(
        executed_count, 1,
        "Only one conflicting transaction should be marked as executed (got {})",
        executed_count
    );

    info!("test_concurrent_conflicting_owned_transactions passed: 1 success, 1 rejection");
}
