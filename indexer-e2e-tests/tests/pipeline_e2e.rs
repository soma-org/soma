// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! TestCluster-based end-to-end tests for the indexer pipeline.
//!
//! These tests start a real Soma network (TestCluster), execute transactions,
//! and verify the indexer processes the resulting checkpoints into Postgres.
//!
//! Requires Postgres on PATH (`brew install postgresql`).
//! Run with: `cargo test -p indexer-e2e-tests --test pipeline_e2e -- --ignored --nocapture`

use std::ops::DerefMut;
use std::time::Duration;

use diesel::prelude::*;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::*;
use indexer_e2e_tests::OffchainCluster;
use indexer_framework::IndexerArgs;
use test_cluster::TestClusterBuilder;
use types::effects::TransactionEffectsAPI;
use types::transaction::{TransactionData, TransactionKind};

/// Verify that a coin transfer transaction is indexed across all relevant pipelines.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_transfer_coin_indexed() {
    let _ = tracing_subscriber::fmt::try_init();

    let ingestion_dir = tempfile::tempdir().unwrap();
    let ingestion_path = ingestion_dir.path().to_path_buf();

    // Start a real network
    let test_cluster =
        TestClusterBuilder::new().with_data_ingestion_dir(ingestion_path.clone()).build().await;

    // Start the off-chain indexer stack
    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(&ingestion_path, IndexerArgs::default(), &registry)
        .await
        .expect("Failed to start OffchainCluster");

    // Execute a TransferCoin transaction
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];
    let gas =
        test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap().unwrap();

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(1000), recipient },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());
    let expected_digest = *response.effects.transaction_digest();

    // Wait for the indexer to catch up (generous timeout for TestCluster startup)
    cluster
        .wait_for_indexer(1, Duration::from_secs(120))
        .await
        .expect("Indexer did not reach checkpoint 1 within timeout");

    let mut conn = cluster.db().connect().await.unwrap();

    // Verify kv_transactions has the transaction
    let tx_count: i64 = kv_transactions::table
        .filter(kv_transactions::tx_digest.eq(expected_digest.inner().to_vec()))
        .count()
        .get_result(conn.deref_mut())
        .await
        .unwrap();
    assert!(tx_count > 0, "Transaction not found in kv_transactions");

    // Verify tx_digests has entries
    let digest_count: i64 = tx_digests::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert!(digest_count > 0, "No entries in tx_digests");

    // Verify kv_checkpoints has at least 1 row
    let cp_count: i64 = kv_checkpoints::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert!(cp_count > 0, "No entries in kv_checkpoints");

    // Verify cp_sequence_numbers has rows
    let seq_count: i64 =
        cp_sequence_numbers::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert!(seq_count > 0, "No entries in cp_sequence_numbers");

    // Verify tx_affected_addresses has sender
    let addr_count: i64 = tx_affected_addresses::table
        .filter(tx_affected_addresses::sender.eq(sender.to_vec()))
        .count()
        .get_result(conn.deref_mut())
        .await
        .unwrap();
    assert!(addr_count > 0, "Sender not found in tx_affected_addresses");

    // Verify tx_balance_changes has entries
    let bal_count: i64 =
        tx_balance_changes::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert!(bal_count > 0, "No entries in tx_balance_changes");

    tracing::info!("test_transfer_coin_indexed passed");
}

/// Verify that watermarks advance correctly across multiple checkpoints.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_watermarks_advance() {
    let _ = tracing_subscriber::fmt::try_init();

    let ingestion_dir = tempfile::tempdir().unwrap();
    let ingestion_path = ingestion_dir.path().to_path_buf();

    let test_cluster =
        TestClusterBuilder::new().with_data_ingestion_dir(ingestion_path.clone()).build().await;

    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(&ingestion_path, IndexerArgs::default(), &registry)
        .await
        .expect("Failed to start OffchainCluster");

    // Execute 3 transactions to generate multiple checkpoints
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];

    for _ in 0..3 {
        let gas =
            test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap().unwrap();

        let tx_data = TransactionData::new(
            TransactionKind::TransferCoin { coin: gas, amount: Some(100), recipient },
            sender,
            vec![gas],
        );

        let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
        assert!(response.effects.status().is_ok());

        // Small delay to encourage different checkpoints
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Wait for the indexer to process at least checkpoint 2
    cluster
        .wait_for_indexer(2, Duration::from_secs(120))
        .await
        .expect("Indexer did not reach checkpoint 2 within timeout");

    // Verify watermarks advanced
    let latest = cluster.latest_checkpoint().await.unwrap();
    assert!(
        latest.is_some() && latest.unwrap() >= 2,
        "Watermarks should be >= 2, got {:?}",
        latest
    );

    // Verify cp_sequence_numbers has multiple rows
    let mut conn = cluster.db().connect().await.unwrap();
    let seq_count: i64 =
        cp_sequence_numbers::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert!(seq_count >= 3, "Expected >= 3 cp_sequence_numbers rows, got {}", seq_count);

    tracing::info!("test_watermarks_advance passed");
}

/// Verify that object-related pipelines index data correctly.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_objects_indexed() {
    let _ = tracing_subscriber::fmt::try_init();

    let ingestion_dir = tempfile::tempdir().unwrap();
    let ingestion_path = ingestion_dir.path().to_path_buf();

    let test_cluster =
        TestClusterBuilder::new().with_data_ingestion_dir(ingestion_path.clone()).build().await;

    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(&ingestion_path, IndexerArgs::default(), &registry)
        .await
        .expect("Failed to start OffchainCluster");

    // Execute a transaction that creates/mutates objects
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];
    let gas =
        test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap().unwrap();

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(500), recipient },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    cluster
        .wait_for_indexer(1, Duration::from_secs(120))
        .await
        .expect("Indexer did not reach checkpoint 1");

    let mut conn = cluster.db().connect().await.unwrap();

    // Verify kv_objects has rows
    let obj_count: i64 = kv_objects::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert!(obj_count > 0, "No entries in kv_objects");

    // Verify obj_versions has rows
    let ver_count: i64 = obj_versions::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert!(ver_count > 0, "No entries in obj_versions");

    // Verify obj_info has rows
    let info_count: i64 = obj_info::table.count().get_result(conn.deref_mut()).await.unwrap();
    assert!(info_count > 0, "No entries in obj_info");

    tracing::info!("test_objects_indexed passed");
}

/// Verify the full stack: TestCluster → Indexer → Postgres → GraphQL.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_graphql_queries_indexed_data() {
    let _ = tracing_subscriber::fmt::try_init();

    let ingestion_dir = tempfile::tempdir().unwrap();
    let ingestion_path = ingestion_dir.path().to_path_buf();

    let test_cluster =
        TestClusterBuilder::new().with_data_ingestion_dir(ingestion_path.clone()).build().await;

    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(&ingestion_path, IndexerArgs::default(), &registry)
        .await
        .expect("Failed to start OffchainCluster");

    // Execute a transaction
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];
    let gas =
        test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap().unwrap();

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(500), recipient },
        sender,
        vec![gas],
    );

    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    // Wait for GraphQL to be queryable
    cluster
        .wait_for_graphql(0, Duration::from_secs(120))
        .await
        .expect("GraphQL did not become ready");

    // Query checkpoint via GraphQL
    let client = reqwest::Client::new();
    let resp = client
        .post(&cluster.graphql_url())
        .json(&serde_json::json!({
            "query": "{ checkpoint { sequenceNumber epoch } }"
        }))
        .send()
        .await
        .expect("GraphQL request failed");

    let json: serde_json::Value = resp.json().await.expect("Failed to parse response");
    let seq = &json["data"]["checkpoint"]["sequenceNumber"];
    assert!(seq.is_string(), "Expected sequenceNumber, got: {}", json);

    // Query service config (should always work)
    let resp = client
        .post(&cluster.graphql_url())
        .json(&serde_json::json!({
            "query": "{ serviceConfig { maxPageSize defaultPageSize } }"
        }))
        .send()
        .await
        .expect("GraphQL request failed");

    let json: serde_json::Value = resp.json().await.expect("Failed to parse response");
    assert!(
        json["data"]["serviceConfig"]["maxPageSize"].is_number(),
        "Expected maxPageSize, got: {}",
        json
    );

    tracing::info!("test_graphql_queries_indexed_data passed");
}

/// Verify that epoch boundary data is indexed correctly.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_epoch_boundary_indexed() {
    let _ = tracing_subscriber::fmt::try_init();

    let ingestion_dir = tempfile::tempdir().unwrap();
    let ingestion_path = ingestion_dir.path().to_path_buf();

    let test_cluster = TestClusterBuilder::new()
        .with_epoch_duration_ms(10_000)
        .with_data_ingestion_dir(ingestion_path.clone())
        .build()
        .await;

    let registry = prometheus::Registry::new();
    let cluster = OffchainCluster::new(&ingestion_path, IndexerArgs::default(), &registry)
        .await
        .expect("Failed to start OffchainCluster");

    // Execute a transaction to help the network progress
    let addresses = test_cluster.wallet.get_addresses();
    let sender = addresses[0];
    let recipient = addresses[1];
    let gas =
        test_cluster.wallet.get_one_gas_object_owned_by_address(sender).await.unwrap().unwrap();

    let tx_data = TransactionData::new(
        TransactionKind::TransferCoin { coin: gas, amount: Some(100), recipient },
        sender,
        vec![gas],
    );
    let _ = test_cluster.sign_and_execute_transaction(&tx_data).await;

    // Wait for epoch boundary — this may take a while with 10s epoch duration.
    // The genesis checkpoint (seq 0) is always an epoch-start, so kv_epoch_starts
    // should have epoch 0 once checkpoint 0 is processed.
    cluster
        .wait_for_indexer(0, Duration::from_secs(120))
        .await
        .expect("Indexer did not process genesis checkpoint");

    let mut conn = cluster.db().connect().await.unwrap();

    // Genesis checkpoint (seq 0) should produce an epoch start entry for epoch 0
    let epoch_start_count: i64 = kv_epoch_starts::table
        .filter(kv_epoch_starts::epoch.eq(0i64))
        .count()
        .get_result(conn.deref_mut())
        .await
        .unwrap();
    assert!(epoch_start_count > 0, "Expected kv_epoch_starts entry for epoch 0");

    // Wait longer for epoch to actually end (epoch_duration_ms = 10s)
    // Poll for epoch 0 end entry
    let epoch_end_found = tokio::time::timeout(Duration::from_secs(180), async {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            let mut c = cluster.db().connect().await.unwrap();
            let count: i64 = kv_epoch_ends::table
                .filter(kv_epoch_ends::epoch.eq(0i64))
                .count()
                .get_result(c.deref_mut())
                .await
                .unwrap();
            if count > 0 {
                break;
            }
        }
    })
    .await;

    assert!(epoch_end_found.is_ok(), "kv_epoch_ends entry for epoch 0 not found within timeout");

    tracing::info!("test_epoch_boundary_indexed passed");
}
