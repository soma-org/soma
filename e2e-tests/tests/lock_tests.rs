// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for object lock behavior after transaction failures.
//!
//! Tests:
//! 1. test_lock_persists_after_insufficient_gas - Demonstrates that a failed
//!    InsufficientGas transaction permanently locks the gas coin within the epoch
//!    on a 3-validator network.

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
use types::supported_protocol_versions::{ProtocolVersion, SupportedProtocolVersions};
use types::system_state::SystemStateTrait as _;
use types::tensor::SomaTensor;
use types::transaction::{SubmitDataArgs, TransactionData, TransactionKind};
use url::Url;
use utils::logging::init_tracing;

// ===================================================================
// Test 1: Lock persists after InsufficientGas failure
//
// On a 3-validator network (quorum = 2), when a transaction fails during
// gas preparation (InsufficientGas), the gas coin's lock is never released
// because error_result() creates an empty InnerTemporaryStore that discards
// gas coin mutations. The gas coin version never advances, so subsequent
// transactions using that coin hit ObjectLockConflict.
//
// Steps:
// 1. Transfer a tiny amount (10 shannons) to create a dust coin
// 2. Use the dust coin as gas for a TransferCoin → fails InsufficientGas
// 3. Attempt another tx with the same dust coin → ObjectLockConflict (bug)
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_lock_persists_after_insufficient_gas() {
    init_tracing();

    // Use 3 validators — quorum is 2, so 2 locked validators = permanent lock
    let test_cluster = TestClusterBuilder::new().with_num_validators(3).build().await;

    let addresses = test_cluster.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    // Step 1: Get a gas coin and transfer a tiny amount to create a dust coin
    let gas_coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(sender, Some(2)).await.unwrap();
    assert!(gas_coins.len() >= 2, "Sender needs at least 2 coins");

    let main_coin = gas_coins[0];
    let funding_gas = gas_coins[1];

    info!("Main coin: {} v{}", main_coin.0, main_coin.1.value());
    info!("Funding gas: {} v{}", funding_gas.0, funding_gas.1.value());

    // Transfer 10 shannons to sender's own address to create a dust coin
    // We use the main_coin as the transfer coin with amount=10, and funding_gas as gas
    let dust_amount: u64 = 10; // 10 shannons — well below any base fee
    let create_dust_tx = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: main_coin,
            amount: Some(dust_amount),
            recipient: sender, // Send to self to create a new small coin
        },
        sender,
        vec![funding_gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&create_dust_tx).await;
    assert!(
        response.effects.status().is_ok(),
        "Dust coin creation should succeed: {:?}",
        response.effects.status()
    );

    // Find the created dust coin (the new coin sent to sender with dust_amount)
    let created = response.effects.created();
    info!("Created {} objects", created.len());

    // The created coin is the one owned by sender (there should be exactly one created)
    let dust_coin_id = created
        .iter()
        .find_map(|(oref, owner)| {
            if matches!(owner, types::object::Owner::AddressOwner(addr) if *addr == sender) {
                Some(oref)
            } else {
                None
            }
        })
        .expect("Should have created a dust coin for sender");

    info!("Dust coin created: {} v{}", dust_coin_id.0, dust_coin_id.1.value());

    // Get an updated ref for the funding gas coin (its version changed)
    let updated_coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(sender, Some(10)).await.unwrap();

    let updated_funding_gas = updated_coins
        .iter()
        .find(|c| c.0 == funding_gas.0)
        .expect("Funding gas should still exist");

    info!("Updated funding gas: {} v{}", updated_funding_gas.0, updated_funding_gas.1.value());

    // Step 2: Use the dust coin as gas for a transaction — should fail with InsufficientGas
    // The dust coin has only 10 shannons, which is below the base fee
    let fail_tx = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: *dust_coin_id, // Use dust coin as the transfer coin AND gas
            amount: Some(1),     // Try to transfer 1 shannon
            recipient,
        },
        sender,
        vec![*dust_coin_id], // Use dust coin as gas payment
    );

    let fail_tx_signed = test_cluster.wallet.sign_transaction(&fail_tx).await;
    let fail_result = test_cluster.wallet.execute_transaction_may_fail(fail_tx_signed).await;

    match &fail_result {
        Ok(resp) => {
            info!("First dust tx completed with effects status: {:?}", resp.effects.status());
            // Should have InsufficientGas in effects
            assert!(resp.effects.status().is_err(), "Transaction with dust gas should fail");
        }
        Err(e) => {
            info!("First dust tx failed at orchestrator level: {}", e);
        }
    }

    // Step 3: Verify the fix — the dust coin should have been properly mutated
    // (version advanced) by into_effects(). Since the partial fee deduction took
    // all 10 shannons, the coin was deleted (balance → 0). This means:
    // - The object version advanced (lock on old version is consumed)
    // - The coin no longer exists (get_gas_objects won't find it)
    //
    // If the bug persists (error_result discards mutations), the dust coin would
    // still exist at the same version with the lock blocking any new transaction.

    let updated_coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(sender, Some(20)).await.unwrap();

    let dust_still_exists = updated_coins.iter().any(|c| c.0 == dust_coin_id.0);

    if dust_still_exists {
        // Dust coin still exists — it may have been partially deducted but not fully consumed.
        // Try using it in a DIFFERENT transaction to verify no ObjectLockConflict.
        let current_dust_ref = updated_coins.iter().find(|c| c.0 == dust_coin_id.0).unwrap();

        info!(
            "Dust coin still exists after failed tx: {} v{} (was v{})",
            current_dust_ref.0,
            current_dust_ref.1.value(),
            dust_coin_id.1.value()
        );

        // Version should have advanced if the fix works
        assert!(
            current_dust_ref.1.value() > dust_coin_id.1.value(),
            "Dust coin version should have advanced after gas preparation failure. \
             Old: v{}, Current: v{}. This indicates error_result() discarded gas mutations.",
            dust_coin_id.1.value(),
            current_dust_ref.1.value()
        );

        // Try a different transaction with the updated ref — should not hit ObjectLockConflict
        let retry_tx = TransactionData::new(
            TransactionKind::TransferCoin {
                coin: *current_dust_ref,
                amount: Some(1),
                recipient: sender,
            },
            sender,
            vec![*current_dust_ref],
        );

        let retry_tx_signed = test_cluster.wallet.sign_transaction(&retry_tx).await;
        let retry_result = test_cluster.wallet.execute_transaction_may_fail(retry_tx_signed).await;

        match &retry_result {
            Ok(resp) => {
                info!("Retry tx effects status: {:?}", resp.effects.status());
            }
            Err(e) => {
                let err_str = format!("{}", e);
                assert!(
                    !err_str.contains("ObjectLockConflict") && !err_str.contains("already locked"),
                    "BUG: Dust coin still locked after fix. Error: {}",
                    e
                );
                info!("Retry tx failed with non-lock error (acceptable): {}", e);
            }
        }
    } else {
        // Dust coin was deleted (balance went to 0 after partial fee deduction).
        // This is the expected outcome: into_effects() preserved the gas coin mutations,
        // the version advanced, and the coin was consumed. The old lock is irrelevant.
        info!(
            "SUCCESS: Dust coin {} was properly consumed (deleted after partial gas deduction). \
             Lock released via version advancement.",
            dust_coin_id.0
        );
    }
}

// ===================================================================
// Test 2: Bond coin lock before and after protocol upgrade
//
// Phase 1 (protocol v1 — bug):
//   A failed SubmitData (TargetNotOpen) does NOT bump the bond coin
//   version, so the epoch-store lock persists and the coin cannot be
//   reused (ObjectLockConflict).
//
// Phase 2 (protocol v2 — fix):
//   After upgrading to v2 (execution_version 1), the same scenario
//   bumps the bond coin version via ensure_active_inputs_mutated(),
//   releasing the lock so the coin is immediately reusable.
// ===================================================================

fn make_manifest(url_str: &str) -> Manifest {
    let url = Url::parse(url_str).expect("Invalid URL");
    let metadata = Metadata::V1(MetadataV1::new(Checksum::new_from_hash([1u8; 32]), 1024));
    Manifest::V1(ManifestV1::new(url, metadata))
}

fn make_submission_manifest(data_size: usize) -> SubmissionManifest {
    let url = Url::parse("https://example.com/data").unwrap();
    let metadata = Metadata::V1(MetadataV1::new(Checksum::new_from_hash([0u8; 32]), data_size));
    let manifest = Manifest::V1(ManifestV1::new(url, metadata));
    SubmissionManifest::new(manifest)
}

/// Helper: fill a target then submit again to get TargetNotOpen.
/// Returns (bond_coin_ref_before, gas_coin_ref_before) used in the failing tx.
async fn fill_target_then_fail(
    test_cluster: &test_cluster::TestCluster,
    submitter: SomaAddress,
    target_id: ObjectID,
    model_id: ObjectID,
    embedding_dim: usize,
    distance_threshold: f32,
) -> (
    (ObjectID, types::object::Version, types::digests::ObjectDigest),
    (ObjectID, types::object::Version, types::digests::ObjectDigest),
) {
    // Fill the target with a successful SubmitData
    let gas1 = test_cluster
        .wallet
        .get_one_gas_object_owned_by_address(submitter)
        .await
        .unwrap()
        .expect("Need gas");

    let fill_tx = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold - 0.1),
            loss_score: SomaTensor::new(vec![0.5], vec![1]),
            bond_coin: gas1,
        }),
        submitter,
        vec![gas1],
    );
    let fill_response = test_cluster.sign_and_execute_transaction(&fill_tx).await;
    assert!(
        fill_response.effects.status().is_ok(),
        "Fill SubmitData should succeed: {:?}",
        fill_response.effects.status()
    );
    info!("Target {} filled successfully", target_id);

    // Get fresh coins for the failing submission
    let coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(submitter, Some(10)).await.unwrap();
    assert!(coins.len() >= 2, "Need at least 2 coins");

    let bond_coin = coins[0];
    let gas_coin = coins[1];
    info!(
        "Bond coin: {} v{}, Gas coin: {} v{}",
        bond_coin.0,
        bond_coin.1.value(),
        gas_coin.0,
        gas_coin.1.value()
    );

    // Submit to the filled target → should fail (TargetNotOpen)
    let fail_tx = TransactionData::new(
        TransactionKind::SubmitData(SubmitDataArgs {
            target_id,
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(distance_threshold - 0.1),
            loss_score: SomaTensor::new(vec![0.5], vec![1]),
            bond_coin,
        }),
        submitter,
        vec![gas_coin],
    );
    let fail_response = test_cluster.sign_and_execute_transaction(&fail_tx).await;
    assert!(fail_response.effects.status().is_err(), "SubmitData to filled target should fail");
    info!("SubmitData to filled target failed as expected");

    (bond_coin, gas_coin)
}

#[cfg(msim)]
#[msim::sim_test]
async fn test_bond_coin_lock_released_after_protocol_upgrade() {
    init_tracing();

    // Create a genesis model so targets are generated
    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let initial_stake = 5 * SHANNONS_PER_SOMA;
    let model_config = GenesisModelConfig {
        owner: model_owner,
        manifest: make_manifest("https://example.com/models/genesis"),
        decryption_key: DecryptionKey::new([0xAA; 32]),
        weights_commitment: ModelWeightsCommitment::new([0xBB; 32]),
        architecture_version: 1,
        commission_rate: 0,
        initial_stake,
    };

    // Start at protocol v1, support up to v2. Use a long epoch so phase 1
    // completes before the epoch boundary triggers the upgrade.
    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(30_000)
        .with_genesis_models(vec![model_config])
        .with_protocol_version(ProtocolVersion::new(1))
        .with_supported_protocol_versions(SupportedProtocolVersions::new_for_testing(1, 2))
        .build()
        .await;
    info!("Cluster started at protocol v1");

    let submitter = test_cluster.get_addresses()[0];
    let client = test_cluster.wallet.get_client().await.unwrap();

    // Read model/target parameters
    let system_state = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_system_state_object_for_testing().expect("SystemState"));
    let model_id = *system_state
        .model_registry()
        .active_models()
        .next()
        .expect("Should have an active model")
        .0;
    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // =====================================================================
    // Phase 1: Protocol v1 — demonstrate the bug
    // =====================================================================

    // Find an open target at epoch 0
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.epoch_filter = Some(0);
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    assert!(!response.targets.is_empty(), "Should have open targets at epoch 0");

    let v1_target_id: ObjectID = response.targets[0]
        .id
        .as_ref()
        .and_then(|id| id.parse().ok())
        .expect("Target should have valid ID");
    info!("[v1] Found open target {}", v1_target_id);

    // Fill the target, then submit again → TargetNotOpen
    let (v1_bond_coin, v1_gas_coin) = fill_target_then_fail(
        &test_cluster,
        submitter,
        v1_target_id,
        model_id,
        embedding_dim,
        distance_threshold,
    )
    .await;

    // Verify we're still in epoch 0 (locks are per-epoch)
    let sys = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_system_state_object_for_testing().expect("SystemState"));
    assert_eq!(sys.epoch(), 0, "Phase 1 must complete within epoch 0");

    // Bond coin version should NOT have advanced (v1 bug)
    let updated_coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(submitter, Some(20)).await.unwrap();
    let v1_updated_bond =
        updated_coins.iter().find(|c| c.0 == v1_bond_coin.0).expect("Bond coin should still exist");

    assert_eq!(
        v1_updated_bond.1.value(),
        v1_bond_coin.1.value(),
        "[v1] Bond coin version should NOT advance: was v{}, now v{}",
        v1_bond_coin.1.value(),
        v1_updated_bond.1.value(),
    );
    info!("[v1] Bond coin version unchanged (bug confirmed): v{}", v1_bond_coin.1.value());

    // Reusing the bond coin should fail with ObjectLockConflict
    let v1_updated_gas =
        updated_coins.iter().find(|c| c.0 == v1_gas_coin.0).expect("Gas coin should still exist");

    let retry_tx = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: *v1_updated_bond,
            amount: Some(1000),
            recipient: submitter,
        },
        submitter,
        vec![*v1_updated_gas],
    );
    let retry_signed = test_cluster.wallet.sign_transaction(&retry_tx).await;
    let retry_result = test_cluster.wallet.execute_transaction_may_fail(retry_signed).await;

    match &retry_result {
        Ok(resp) => {
            panic!(
                "[v1] Retry should have failed with ObjectLockConflict, but got: {:?}",
                resp.effects.status()
            );
        }
        Err(e) => {
            let err_str = format!("{}", e);
            assert!(
                err_str.contains("already locked") || err_str.contains("ObjectLockConflict"),
                "[v1] Expected ObjectLockConflict but got: {}",
                e
            );
            info!("[v1] Bond coin locked as expected (ObjectLockConflict)");
        }
    }

    // =====================================================================
    // Phase 2: Protocol v2 — demonstrate the fix
    // =====================================================================

    let target_version = ProtocolVersion::new(2);
    let system_state = test_cluster.wait_for_protocol_version(target_version).await;
    let current_epoch = system_state.epoch();
    assert_eq!(system_state.protocol_version(), 2);
    info!("[v2] Protocol upgraded to v2 at epoch {}", current_epoch);

    // Re-read parameters (may change across epochs)
    let system_state = test_cluster
        .fullnode_handle
        .soma_node
        .with(|node| node.state().get_system_state_object_for_testing().expect("SystemState"));
    let embedding_dim = system_state.parameters().target_embedding_dim as usize;
    let distance_threshold = system_state.target_state().distance_threshold.as_scalar();

    // Find an open target from the current epoch
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.epoch_filter = Some(current_epoch);
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    assert!(!response.targets.is_empty(), "Should have open targets in epoch {}", current_epoch);

    let v2_target_id: ObjectID = response.targets[0]
        .id
        .as_ref()
        .and_then(|id| id.parse().ok())
        .expect("Target should have valid ID");
    info!("[v2] Found open target {}", v2_target_id);

    // Fill the target, then submit again → TargetNotOpen
    let (v2_bond_coin, v2_gas_coin) = fill_target_then_fail(
        &test_cluster,
        submitter,
        v2_target_id,
        model_id,
        embedding_dim,
        distance_threshold,
    )
    .await;

    // Bond coin version SHOULD have advanced (v2 fix)
    let updated_coins =
        test_cluster.wallet.get_gas_objects_owned_by_address(submitter, Some(20)).await.unwrap();
    let v2_updated_bond =
        updated_coins.iter().find(|c| c.0 == v2_bond_coin.0).expect("Bond coin should still exist");

    assert!(
        v2_updated_bond.1.value() > v2_bond_coin.1.value(),
        "[v2] Bond coin version should advance: was v{}, now v{}",
        v2_bond_coin.1.value(),
        v2_updated_bond.1.value(),
    );
    info!(
        "[v2] Bond coin version advanced: v{} -> v{}",
        v2_bond_coin.1.value(),
        v2_updated_bond.1.value()
    );

    // Reusing the bond coin should succeed (no ObjectLockConflict)
    let v2_updated_gas =
        updated_coins.iter().find(|c| c.0 == v2_gas_coin.0).expect("Gas coin should still exist");

    let retry_tx = TransactionData::new(
        TransactionKind::TransferCoin {
            coin: *v2_updated_bond,
            amount: Some(1000),
            recipient: submitter,
        },
        submitter,
        vec![*v2_updated_gas],
    );
    let retry_signed = test_cluster.wallet.sign_transaction(&retry_tx).await;
    let retry_result = test_cluster.wallet.execute_transaction_may_fail(retry_signed).await;

    match &retry_result {
        Ok(resp) => {
            info!("[v2] Retry tx completed: {:?}", resp.effects.status());
        }
        Err(e) => {
            let err_str = format!("{}", e);
            assert!(
                !err_str.contains("ObjectLockConflict") && !err_str.contains("already locked"),
                "[v2] BUG: Bond coin still locked after version bump. Error: {}",
                e
            );
            info!("[v2] Retry tx failed with non-lock error (acceptable): {}", e);
        }
    }

    info!("test_bond_coin_lock_released_after_protocol_upgrade passed");
}
