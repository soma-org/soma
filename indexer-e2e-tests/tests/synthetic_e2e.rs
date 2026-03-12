// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! TestCheckpointBuilder-based end-to-end tests for Soma-specific indexer pipelines.
//!
//! These tests create synthetic checkpoints (with targets, rewards, etc.) that
//! TestCluster cannot produce, write them to disk, and verify the indexer
//! processes them correctly through the real pipeline framework.
//!
//! Requires Postgres on PATH (`brew install postgresql`).
//! Run with: `cargo test -p indexer-e2e-tests --test synthetic_e2e -- --ignored --nocapture`

use std::ops::DerefMut;
use std::time::Duration;

use diesel::prelude::*;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::*;
use indexer_e2e_tests::{OffchainCluster, write_checkpoint_file};
use indexer_framework::IndexerArgs;
use types::base::SomaAddress;
use types::target::TargetStatus;
use types::test_checkpoint_data_builder::{TestCheckpointBuilder, default_test_system_state, test_target, test_filled_target};

/// Verify soma_targets pipeline works through the real indexer framework.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_soma_targets_through_pipeline() {
    let _ = tracing_subscriber::fmt::try_init();

    let checkpoint_dir = tempfile::tempdir().unwrap();
    let dir = checkpoint_dir.path();

    // Build a checkpoint with a Target object and genesis system state
    let target = test_target(0, TargetStatus::Open, 5_000_000);
    let checkpoint = TestCheckpointBuilder::new(0)
        .with_epoch(0)
        .with_network_total_transactions(1)
        .with_genesis_system_state(default_test_system_state())
        .add_target(target)
        .build();

    write_checkpoint_file(dir, &checkpoint);

    // Start the indexer pointed at this directory
    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(
        dir,
        IndexerArgs {
            last_checkpoint: Some(0),
            ..Default::default()
        },
        &registry,
    )
    .await
    .expect("Failed to start OffchainCluster");

    cluster
        .wait_for_indexer(0, Duration::from_secs(60))
        .await
        .expect("Indexer did not process checkpoint 0");

    let mut conn = cluster.db().connect().await.unwrap();

    // Verify soma_targets has a row
    let target_count: i64 = soma_targets::table
        .count()
        .get_result(conn.deref_mut())
        .await
        .unwrap();
    assert!(
        target_count > 0,
        "Expected at least 1 row in soma_targets"
    );

    // Verify the target has the correct status
    let statuses: Vec<String> = soma_targets::table
        .select(soma_targets::status)
        .load(conn.deref_mut())
        .await
        .unwrap();
    assert!(
        statuses.contains(&"open".to_string()),
        "Expected an open target, got: {:?}",
        statuses
    );

    // Verify reward_pool
    let pools: Vec<i64> = soma_targets::table
        .select(soma_targets::reward_pool)
        .load(conn.deref_mut())
        .await
        .unwrap();
    assert!(
        pools.contains(&5_000_000),
        "Expected reward_pool = 5000000, got: {:?}",
        pools
    );

    tracing::info!("test_soma_targets_through_pipeline passed");
}

/// Verify soma_rewards pipeline works through the real indexer framework.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_soma_rewards_through_pipeline() {
    let _ = tracing_subscriber::fmt::try_init();

    let checkpoint_dir = tempfile::tempdir().unwrap();
    let dir = checkpoint_dir.path();

    let sender = SomaAddress::random();
    let target_id = types::object::ObjectID::random();

    let checkpoint = TestCheckpointBuilder::new(0)
        .with_epoch(0)
        .with_network_total_transactions(1)
        .with_genesis_system_state(default_test_system_state())
        .add_claim_rewards(sender, target_id, 1_000_000)
        .build();

    write_checkpoint_file(dir, &checkpoint);

    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(
        dir,
        IndexerArgs {
            last_checkpoint: Some(0),
            ..Default::default()
        },
        &registry,
    )
    .await
    .expect("Failed to start OffchainCluster");

    cluster
        .wait_for_indexer(0, Duration::from_secs(60))
        .await
        .expect("Indexer did not process checkpoint 0");

    let mut conn = cluster.db().connect().await.unwrap();

    // Verify soma_rewards has a row
    let reward_count: i64 = soma_rewards::table
        .count()
        .get_result(conn.deref_mut())
        .await
        .unwrap();
    assert!(
        reward_count > 0,
        "Expected at least 1 row in soma_rewards"
    );

    // Verify the target_id matches
    let target_ids: Vec<Vec<u8>> = soma_rewards::table
        .select(soma_rewards::target_id)
        .load(conn.deref_mut())
        .await
        .unwrap();
    assert!(
        target_ids.contains(&target_id.to_vec()),
        "Expected target_id in soma_rewards"
    );

    tracing::info!("test_soma_rewards_through_pipeline passed");
}

/// Verify target lifecycle across multiple checkpoints through the pipeline.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_multi_checkpoint_with_targets() {
    let _ = tracing_subscriber::fmt::try_init();

    let checkpoint_dir = tempfile::tempdir().unwrap();
    let dir = checkpoint_dir.path();

    let target_id = types::object::ObjectID::random();
    let submitter = SomaAddress::random();
    let model_id = types::object::ObjectID::random();

    // Checkpoint 0: Target created as Open (with genesis system state for kv_epoch_starts/soma_models)
    let open_target = test_target(0, TargetStatus::Open, 5_000_000);
    let cp0 = TestCheckpointBuilder::new(0)
        .with_epoch(0)
        .with_network_total_transactions(1)
        .with_genesis_system_state(default_test_system_state())
        .add_target_with_id(target_id, open_target)
        .build();

    // Checkpoint 1: Target filled
    let filled_target = test_filled_target(0, 0, submitter, model_id, 5_000_000, 1_000_000);
    let cp1 = TestCheckpointBuilder::new(1)
        .with_epoch(0)
        .with_network_total_transactions(2)
        .add_target_with_id(target_id, filled_target)
        .build();

    write_checkpoint_file(dir, &cp0);
    write_checkpoint_file(dir, &cp1);

    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(
        dir,
        IndexerArgs {
            last_checkpoint: Some(1),
            ..Default::default()
        },
        &registry,
    )
    .await
    .expect("Failed to start OffchainCluster");

    cluster
        .wait_for_indexer(1, Duration::from_secs(60))
        .await
        .expect("Indexer did not reach checkpoint 1");

    let mut conn = cluster.db().connect().await.unwrap();

    // Verify soma_targets has 2 rows for this target (one Open, one Filled)
    let target_rows: Vec<(Vec<u8>, String)> = soma_targets::table
        .filter(soma_targets::target_id.eq(target_id.to_vec()))
        .select((soma_targets::target_id, soma_targets::status))
        .order(soma_targets::cp_sequence_number.asc())
        .load(conn.deref_mut())
        .await
        .unwrap();

    assert_eq!(
        target_rows.len(),
        2,
        "Expected 2 target versions, got {}",
        target_rows.len()
    );
    assert_eq!(target_rows[0].1, "open");
    assert_eq!(target_rows[1].1, "filled");

    // Verify watermarks are all at checkpoint 1
    let latest = cluster.latest_checkpoint().await.unwrap();
    assert_eq!(
        latest,
        Some(1),
        "All watermarks should be at checkpoint 1"
    );

    tracing::info!("test_multi_checkpoint_with_targets passed");
}

/// Verify the indexer service exits cleanly after processing the last checkpoint.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_indexer_exits_after_last_checkpoint() {
    let _ = tracing_subscriber::fmt::try_init();

    let checkpoint_dir = tempfile::tempdir().unwrap();
    let dir = checkpoint_dir.path();

    // Simple checkpoint with genesis system state
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let checkpoint = TestCheckpointBuilder::new(0)
        .with_epoch(0)
        .with_network_total_transactions(1)
        .with_genesis_system_state(default_test_system_state())
        .add_transfer_coin(sender, recipient, 100_000)
        .build();

    write_checkpoint_file(dir, &checkpoint);

    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(
        dir,
        IndexerArgs {
            last_checkpoint: Some(0),
            ..Default::default()
        },
        &registry,
    )
    .await
    .expect("Failed to start OffchainCluster");

    cluster
        .wait_for_indexer(0, Duration::from_secs(30))
        .await
        .expect("Indexer did not process checkpoint 0");

    let mut conn = cluster.db().connect().await.unwrap();

    let cp_count: i64 = kv_checkpoints::table
        .count()
        .get_result(conn.deref_mut())
        .await
        .unwrap();
    assert_eq!(cp_count, 1, "Expected exactly 1 checkpoint");

    let tx_count: i64 = tx_digests::table
        .count()
        .get_result(conn.deref_mut())
        .await
        .unwrap();
    assert!(tx_count > 0, "Expected at least 1 tx_digest");

    tracing::info!("test_indexer_exits_after_last_checkpoint passed");
}
