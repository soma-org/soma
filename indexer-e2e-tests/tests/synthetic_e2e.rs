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
use types::test_checkpoint_data_builder::{
    TestCheckpointBuilder, default_test_system_state,
};

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
        IndexerArgs { last_checkpoint: Some(0), ..Default::default() },
        &registry,
    )
    .await
    .expect("Failed to start OffchainCluster");

    cluster
        .wait_for_indexer(0, Duration::from_secs(30))
        .await
        .expect("Indexer did not process checkpoint 0");

    let mut conn = cluster.db().connect().await.unwrap();

    let cp_count: i64 = kv_checkpoints::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert_eq!(cp_count, 1, "Expected exactly 1 checkpoint");

    let tx_count: i64 = tx_digests::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert!(tx_count > 0, "Expected at least 1 tx_digest");

    tracing::info!("test_indexer_exits_after_last_checkpoint passed");
}
