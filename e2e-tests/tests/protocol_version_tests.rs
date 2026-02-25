// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Protocol version upgrade tests.
//!
//! Tests:
//! 1. test_validator_panics_on_unsupported_protocol_version — genesis at unsupported version panics
//! 2. test_protocol_version_upgrade — all validators upgrade from v1 to v2
//! 3. test_protocol_version_upgrade_no_quorum — upgrade fails without 75% quorum
//! 4. test_protocol_version_upgrade_one_laggard — upgrade succeeds with 75% quorum, laggard shuts down
//! 5. test_protocol_version_upgrade_with_shutdown_validator — upgrade succeeds with stopped validator
//! 6. test_protocol_version_upgrade_insufficient_support — 25% support can't upgrade
//!
//! Ported from Sui's `protocol_version_tests.rs`.
//! Skipped: 19 of 26 tests that require Move framework or framework-specific types.

use std::sync::Arc;
use std::time::Duration;
use test_cluster::TestClusterBuilder;
use tokio::time::sleep;
use tracing::info;
use types::{
    supported_protocol_versions::{ProtocolVersion, SupportedProtocolVersions},
    system_state::{SystemStateTrait, epoch_start::EpochStartSystemStateTrait as _},
};
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

/// Validators 0,1 support only v1; validators 2,3 support v1-v2.
/// 50% < 75% quorum, so upgrade should NOT happen.
#[cfg(msim)]
#[msim::sim_test]
async fn test_protocol_version_upgrade_no_quorum() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(20_000)
        .with_supported_protocol_version_callback(Arc::new(|idx, _name| {
            if idx < 2 {
                // Validators 0, 1: only support v1
                SupportedProtocolVersions::new_for_testing(1, 1)
            } else {
                // Validators 2, 3: support v1-v2
                SupportedProtocolVersions::new_for_testing(1, ProtocolVersion::MAX_ALLOWED.as_u64())
            }
        }))
        .build()
        .await;

    // Wait for an epoch transition
    test_cluster.wait_for_epoch(None).await;

    // Protocol version should remain at 1
    let version = test_cluster.highest_protocol_version();
    assert_eq!(version.as_u64(), 1, "Protocol version should remain at 1 without quorum");

    info!("Protocol version correctly stayed at 1 without upgrade quorum");
}

/// Validators 0,1,2 support v1-v2; validator 3 only supports v1.
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
                // Validators 0, 1, 2: support v1-v2
                SupportedProtocolVersions::new_for_testing(1, ProtocolVersion::MAX_ALLOWED.as_u64())
            } else {
                // Validator 3: only supports v1 (laggard)
                SupportedProtocolVersions::new_for_testing(1, 1)
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

/// Only 1/4 validators support v2. Even with buffer_stake=0, 25% < 66.7% quorum,
/// so the upgrade should not happen.
#[cfg(msim)]
#[msim::sim_test]
async fn test_protocol_version_upgrade_insufficient_support() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(20_000)
        .with_supported_protocol_version_callback(Arc::new(|idx, _name| {
            if idx == 0 {
                // Only validator 0 supports v2
                SupportedProtocolVersions::new_for_testing(1, ProtocolVersion::MAX_ALLOWED.as_u64())
            } else {
                SupportedProtocolVersions::new_for_testing(1, 1)
            }
        }))
        .build()
        .await;

    // Wait for an epoch transition
    test_cluster.wait_for_epoch(None).await;

    // Protocol version should remain at 1 — 25% is well below 2/3 quorum
    let version = test_cluster.highest_protocol_version();
    assert_eq!(version.as_u64(), 1, "Protocol version should remain at 1 with only 25% support");

    info!("Protocol version correctly stayed at 1 with insufficient support");
}
