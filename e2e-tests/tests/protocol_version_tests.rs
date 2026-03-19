// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Protocol version upgrade tests.
//!
//! Tests:
//! 1. test_validator_panics_on_unsupported_protocol_version — genesis at unsupported version panics
//! 2. test_protocol_version_upgrade — all validators upgrade from v1 to v2
//! 3. test_protocol_version_upgrade_no_quorum — upgrade to MAX_ALLOWED fails without 75% quorum
//! 4. test_protocol_version_upgrade_one_laggard — upgrade succeeds with 75% quorum, laggard shuts down
//! 5. test_protocol_version_upgrade_with_shutdown_validator — upgrade succeeds with stopped validator
//! 6. test_protocol_version_upgrade_insufficient_support — 25% support can't upgrade
//!
//! Ported from Sui's `protocol_version_tests.rs`.
//! Skipped: 19 of 26 tests that require Move framework or framework-specific types.

use std::sync::Arc;
use std::time::Duration;

use rpc::proto::soma::ListTargetsRequest;
use test_cluster::TestClusterBuilder;
use tokio::time::sleep;
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
use types::system_state::SystemStateTrait;
use types::system_state::epoch_start::EpochStartSystemStateTrait as _;
use types::tensor::SomaTensor;
use types::transaction::{SubmitDataArgs, TransactionData, TransactionKind};
use url::Url;
use utils::logging::init_tracing;

/// Create a cluster at a protocol version beyond MAX_ALLOWED. The cluster
/// should panic because the protocol version config is unsupported.
#[cfg(msim)]
#[msim::sim_test]
#[should_panic]
async fn test_validator_panics_on_unsupported_protocol_version() {
    let _ = tracing_subscriber::fmt::try_init();

    // Version MAX_ALLOWED+1 is always unsupported (even in msim)
    let unsupported = ProtocolVersion::new(ProtocolVersion::MAX_ALLOWED.as_u64() + 1);

    TestClusterBuilder::new()
        .with_protocol_version(unsupported)
        .with_supported_protocol_versions(SupportedProtocolVersions::new_for_testing(
            1,
            unsupported.as_u64(),
        ))
        .build()
        .await;
}

/// All 4 validators support v1-v2. After an epoch transition, the protocol
/// version should upgrade to v2.
#[cfg(msim)]
#[msim::sim_test]
async fn test_protocol_version_upgrade() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(20_000)
        .with_supported_protocol_versions(SupportedProtocolVersions::new_for_testing(
            1,
            ProtocolVersion::MAX_ALLOWED.as_u64(),
        ))
        .build()
        .await;

    let target = ProtocolVersion::MAX_ALLOWED;
    let system_state = test_cluster.wait_for_protocol_version(target).await;
    assert_eq!(
        system_state.protocol_version(),
        target.as_u64(),
        "Protocol version should have upgraded to {}",
        target.as_u64()
    );

    info!("Protocol version upgraded to {} successfully", system_state.protocol_version());
}

/// Validators 0,1 support only up to MAX; validators 2,3 support up to MAX_ALLOWED.
/// 50% < 75% quorum, so upgrade to MAX_ALLOWED should NOT happen.
#[cfg(msim)]
#[msim::sim_test]
async fn test_protocol_version_upgrade_no_quorum() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(20_000)
        .with_supported_protocol_version_callback(Arc::new(|idx, _name| {
            if idx < 2 {
                // Validators 0, 1: only support up to current MAX
                SupportedProtocolVersions::new_for_testing(1, ProtocolVersion::MAX.as_u64())
            } else {
                // Validators 2, 3: support up to MAX_ALLOWED (fake next version)
                SupportedProtocolVersions::new_for_testing(1, ProtocolVersion::MAX_ALLOWED.as_u64())
            }
        }))
        .build()
        .await;

    // Wait for an epoch transition
    test_cluster.wait_for_epoch(None).await;

    // Protocol version should remain at MAX — 50% is below 2/3 quorum for upgrade
    let version = test_cluster.highest_protocol_version();
    assert_eq!(
        version.as_u64(),
        ProtocolVersion::MAX.as_u64(),
        "Protocol version should remain at MAX without quorum"
    );

    info!("Protocol version correctly stayed at {} without upgrade quorum", version.as_u64());
}

/// Validators 0,1,2 support up to MAX_ALLOWED; validator 3 only supports up to MAX.
/// 75% (3/4) exceeds the 2/3 BFT quorum threshold, so upgrade succeeds.
/// (In msim, buffer_stake_for_protocol_upgrade_bps=0, so only 2/3 quorum is needed.)
#[cfg(msim)]
#[msim::sim_test]
async fn test_protocol_version_upgrade_one_laggard() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(20_000)
        .with_supported_protocol_version_callback(Arc::new(|idx, _name| {
            if idx < 3 {
                // Validators 0, 1, 2: support up to MAX_ALLOWED (fake next version)
                SupportedProtocolVersions::new_for_testing(1, ProtocolVersion::MAX_ALLOWED.as_u64())
            } else {
                // Validator 3: only supports up to MAX (laggard)
                SupportedProtocolVersions::new_for_testing(1, ProtocolVersion::MAX.as_u64())
            }
        }))
        .build()
        .await;

    let target = ProtocolVersion::MAX_ALLOWED;
    let system_state = test_cluster.wait_for_protocol_version(target).await;
    assert_eq!(
        system_state.protocol_version(),
        target.as_u64(),
        "Protocol version should have upgraded with 75% quorum"
    );

    info!("Protocol version upgraded to {} with one laggard", system_state.protocol_version());
}

/// All 4 validators support v1-v2. Stop validator 0.
/// 3/4 remaining exceeds 2/3 quorum, so upgrade proceeds.
/// Restart validator 0, verify it catches up.
#[cfg(msim)]
#[msim::sim_test]
async fn test_protocol_version_upgrade_with_shutdown_validator() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(20_000)
        .with_supported_protocol_versions(SupportedProtocolVersions::new_for_testing(
            1,
            ProtocolVersion::MAX_ALLOWED.as_u64(),
        ))
        .build()
        .await;

    // Stop one validator
    let validators = test_cluster.get_validator_pubkeys();
    test_cluster.stop_node(&validators[0]);
    info!("Stopped validator 0");

    // Wait for the upgrade with remaining 3/4 validators
    let target = ProtocolVersion::MAX_ALLOWED;
    let system_state = test_cluster.wait_for_protocol_version(target).await;
    assert_eq!(
        system_state.protocol_version(),
        target.as_u64(),
        "Protocol version should upgrade with 3/4 validators"
    );

    // Restart the stopped validator
    test_cluster.start_node(&validators[0]).await;
    info!("Restarted validator 0");

    // Wait for the restarted validator to catch up
    sleep(Duration::from_secs(5)).await;

    let handle = test_cluster.swarm.node(&validators[0]).unwrap().get_node_handle().unwrap();
    let caught_up_version = handle
        .with(|node| node.state().epoch_store_for_testing().epoch_start_state().protocol_version());
    assert_eq!(
        caught_up_version.as_u64(),
        target.as_u64(),
        "Restarted validator should catch up to new protocol version"
    );

    info!("Validator 0 caught up to protocol version {}", target.as_u64());
}

/// Only 1/4 validators support MAX_ALLOWED. Even with buffer_stake=0, 25% < 66.7% quorum,
/// so the upgrade should not happen.
#[cfg(msim)]
#[msim::sim_test]
async fn test_protocol_version_upgrade_insufficient_support() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(20_000)
        .with_supported_protocol_version_callback(Arc::new(|idx, _name| {
            if idx == 0 {
                // Only validator 0 supports MAX_ALLOWED
                SupportedProtocolVersions::new_for_testing(1, ProtocolVersion::MAX_ALLOWED.as_u64())
            } else {
                // Others only support up to current MAX
                SupportedProtocolVersions::new_for_testing(1, ProtocolVersion::MAX.as_u64())
            }
        }))
        .build()
        .await;

    // Wait for an epoch transition
    test_cluster.wait_for_epoch(None).await;

    // Protocol version should remain at MAX — 25% is well below 2/3 quorum
    let version = test_cluster.highest_protocol_version();
    assert_eq!(
        version.as_u64(),
        ProtocolVersion::MAX.as_u64(),
        "Protocol version should remain at MAX with only 25% support"
    );

    info!("Protocol version correctly stayed at {} with insufficient support", version.as_u64());
}

// ===== V3 upgrade test helpers =====

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
// Test 7: Protocol V3 upgrade — difficulty and emission fix
//
// End-to-end test for the V2→V3 protocol upgrade verifying:
// - V2 parameters (old reward, old threshold, old bond)
// - Successful submission at V2
// - V3 upgrade triggers migration (threshold reset, EMA reset)
// - V3 parameters (new reward per target_hits_per_epoch, higher bond)
// - Successful submission at V3
// - V3 z-based difficulty adjustment eases when hits << target
// - Emission pool does not drain excessively
// ===================================================================

#[cfg(msim)]
#[msim::sim_test]
async fn test_protocol_v3_upgrade_difficulty_and_emission() {
    init_tracing();

    // Create a genesis model so targets are generated
    let model_owner = SomaAddress::from_bytes([0x01; 32]).unwrap();
    let model_config = make_genesis_model_config(model_owner, 5 * SHANNONS_PER_SOMA);

    // Start at V2, validators support up to V3
    let test_cluster = TestClusterBuilder::new()
        .with_protocol_version(ProtocolVersion::new(2))
        .with_supported_protocol_versions(SupportedProtocolVersions::new_for_testing(1, 3))
        .with_genesis_models(vec![model_config])
        .build()
        .await;

    // ===== Phase 1: Verify V2 state =====
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    assert_eq!(system_state.protocol_version(), 2, "Genesis should be at V2");

    // V2 parameters
    assert_eq!(system_state.parameters().target_hits_per_epoch, 16);
    assert_eq!(system_state.parameters().submission_bond_per_byte, 10);
    assert_eq!(system_state.parameters().target_initial_targets_per_epoch, 20);

    // V2 reward_per_target at genesis: uses 2x bootstrap estimate
    // reward = (emission_per_epoch * 80%) / (initial_targets * 2)
    let v2_reward = system_state.target_state().reward_per_target;
    let expected_v2_reward = {
        let e = system_state.emission_pool().emission_per_epoch;
        let d = system_state.parameters().target_initial_targets_per_epoch * 2; // genesis 2x bootstrap
        (e as u128 * 8000 / 10000 / d as u128) as u64
    };
    assert_eq!(
        v2_reward, expected_v2_reward,
        "V2 genesis reward should be emission*80%/(initial_targets*2)"
    );
    // V2 reward should be substantial (billions of shannons)
    assert!(v2_reward > 1_000_000_000, "V2 reward should be in billions, got {}", v2_reward);

    // V2 threshold: initial = 2.0
    let v2_threshold = system_state.target_state().distance_threshold.as_scalar();
    assert!(
        (v2_threshold - 2.0).abs() < 0.01,
        "V2 initial threshold should be 2.0, got {}",
        v2_threshold
    );

    let emission_pool_before = system_state.emission_pool().balance;
    let emission_per_epoch = system_state.emission_pool().emission_per_epoch;

    info!(
        "V2 verified: reward={}, threshold={:.4}, bond={}, emission_pool={}",
        v2_reward,
        v2_threshold,
        system_state.parameters().submission_bond_per_byte,
        emission_pool_before
    );

    // ===== Phase 2: Submit data at V2 =====
    let submitter = test_cluster.get_addresses()[0];
    let model_id = *system_state
        .model_registry()
        .active_models()
        .next()
        .expect("Should have at least one active model")
        .0;
    let embedding_dim = system_state.parameters().target_embedding_dim as usize;

    let client = test_cluster.wallet.get_client().await.unwrap();
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    assert!(!response.targets.is_empty(), "Should have open targets at V2");

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
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(v2_threshold - 0.1),
            loss_score: SomaTensor::new(vec![0.5], vec![1]),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let response = test_cluster.sign_and_execute_transaction(&submit_tx).await;
    assert!(
        response.effects.status().is_ok(),
        "V2 submission should succeed: {:?}",
        response.effects.status()
    );
    info!("V2 submission succeeded");

    // ===== Phase 3: Trigger V3 upgrade =====
    test_cluster.trigger_reconfiguration().await;

    // ===== Phase 4: Verify V3 state =====
    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    assert_eq!(system_state.protocol_version(), 3, "Should have upgraded to V3");

    // V3 parameters
    assert_eq!(
        system_state.parameters().target_hits_per_epoch,
        86_400,
        "V3 target_hits_per_epoch should be 86,400"
    );
    assert_eq!(system_state.parameters().submission_bond_per_byte, 65, "V3 bond should be 65/byte");
    assert_eq!(
        system_state.parameters().target_difficulty_adjustment_rate_bps,
        200,
        "V3 adjustment rate should be 200 bps"
    );
    assert_eq!(
        system_state.parameters().target_hits_ema_decay_bps,
        9000,
        "V3 EMA decay should be 9000 bps"
    );

    // V3 min threshold raised to 0.95
    let v3_min_threshold = system_state.parameters().target_min_distance_threshold.as_scalar();
    assert!(
        (v3_min_threshold - 0.95).abs() < 0.01,
        "V3 min threshold should be 0.95, got {}",
        v3_min_threshold
    );

    // V3 migration: threshold reset near 0.978 (z=1)
    // Note: after migration, adjust_difficulty runs and may shift it slightly
    // if there were hits in the V2 epoch. With 1 hit << 86,400 target, it eases.
    let v3_threshold = system_state.target_state().distance_threshold.as_scalar();
    assert!(
        v3_threshold > 0.97 && v3_threshold < 1.01,
        "V3 threshold should be near 0.978 (may ease slightly), got {}",
        v3_threshold
    );

    // V3 reward_per_target: (emission_per_epoch * 80%) / 86,400
    let v3_reward = system_state.target_state().reward_per_target;
    let expected_v3_reward = {
        let e = system_state.emission_pool().emission_per_epoch;
        (e as u128 * 8000 / 10000 / 86_400) as u64
    };
    assert_eq!(v3_reward, expected_v3_reward, "V3 reward should be emission*80%/86400");
    // V3 reward should be massively smaller than V2
    assert!(
        v3_reward < v2_reward / 100,
        "V3 reward ({}) should be much smaller than V2 reward ({})",
        v3_reward,
        v2_reward
    );

    info!(
        "V3 verified: reward={}, threshold={:.4}, bond={}, min_threshold={:.2}",
        v3_reward,
        v3_threshold,
        system_state.parameters().submission_bond_per_byte,
        v3_min_threshold
    );

    // ===== Phase 5: Submit data at V3 =====
    // Query targets from current epoch (epoch 0 targets are expired)
    let mut request = ListTargetsRequest::default();
    request.status_filter = Some("open".to_string());
    request.epoch_filter = Some(system_state.epoch());
    request.page_size = Some(1);
    let response = client.list_targets(request).await.unwrap();
    assert!(!response.targets.is_empty(), "Should have open targets at V3 epoch");

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
            data_manifest: make_submission_manifest(1024),
            model_id,
            embedding: SomaTensor::zeros(vec![embedding_dim]),
            distance_score: SomaTensor::scalar(v3_threshold - 0.1),
            loss_score: SomaTensor::new(vec![0.5], vec![1]),
            bond_coin: gas_object,
        }),
        submitter,
        vec![gas_object],
    );

    let response = test_cluster.sign_and_execute_transaction(&submit_tx).await;
    assert!(
        response.effects.status().is_ok(),
        "V3 submission should succeed: {:?}",
        response.effects.status()
    );
    info!("V3 submission succeeded");

    // ===== Phase 6: Advance another epoch, verify V3 difficulty adjustment =====
    test_cluster.trigger_reconfiguration().await;

    let system_state = test_cluster.fullnode_handle.soma_node.with(|node| {
        node.state()
            .get_system_state_object_for_testing()
            .expect("Should be able to get SystemState")
    });

    assert_eq!(system_state.protocol_version(), 3);

    // With very few hits (1-2) vs target of 86,400, z-based adjustment should ease
    // (threshold moves toward 1.0, z decreases)
    let post_adj_threshold = system_state.target_state().distance_threshold.as_scalar();
    assert!(
        post_adj_threshold >= v3_threshold - 0.001,
        "With few hits, difficulty should ease or stay. Got {} (was {})",
        post_adj_threshold,
        v3_threshold
    );
    assert!(
        post_adj_threshold <= 1.0,
        "Threshold should not exceed 1.0 (max cosine distance for meaningful embeddings), got {}",
        post_adj_threshold
    );

    info!(
        "V3 difficulty adjusted: threshold {:.4} -> {:.4} (eased as expected with few hits)",
        v3_threshold, post_adj_threshold
    );

    // ===== Phase 7: Verify emission pool is not draining excessively =====
    let emission_after = system_state.emission_pool().balance;
    // Two epochs have passed since genesis. Pool should decrease by at most
    // ~2 * emission_per_epoch (plus a small amount for target rewards from initial targets).
    // The key invariant: with V3's small per-target rewards, the pool should NOT
    // drain significantly faster than the linear emission schedule.
    let max_expected_drain = 3 * emission_per_epoch; // generous headroom
    let actual_drain = emission_pool_before.saturating_sub(emission_after);
    assert!(
        actual_drain <= max_expected_drain,
        "Emission pool drained too fast: {} shannons over 2 epochs (max expected {}). \
         Before={}, after={}",
        actual_drain,
        max_expected_drain,
        emission_pool_before,
        emission_after
    );

    info!(
        "Emission pool healthy: drained {} / {} per_epoch over 2 epochs ({:.1}x linear rate)",
        actual_drain,
        emission_per_epoch,
        actual_drain as f64 / (2.0 * emission_per_epoch as f64)
    );

    info!("test_protocol_v3_upgrade_difficulty_and_emission passed");
}
