// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Indexer integration tests.
//!
//! Tests:
//! 1. test_rpc_ingestion_source — gRPC unary ingestion source reaches the fullnode,
//!    and a fetched checkpoint survives the encode→decode round-trip
//! 2. test_streaming_ingestion_source — gRPC streaming ingestion source

use std::sync::Arc;
use std::time::Duration;

use indexer_framework::ingestion::ingestion_client::IngestionClient;
use indexer_framework::ingestion::ingestion_client::IngestionClientArgs;
use indexer_framework::metrics::IngestionMetrics;
use rpc::utils::checkpoint_blob;
use test_cluster::TestClusterBuilder;
use tokio::time::sleep;
use tracing::info;
use types::effects::TransactionEffectsAPI;
use types::full_checkpoint_content::Checkpoint;
use utils::logging::init_tracing;

/// Verify the gRPC unary ingestion source (RpcIngestionClient, exposed via the
/// `--rpc-api-url` flag) reaches the fullnode's LedgerService and returns decodable
/// checkpoints. This is the Phase 6 reachability check: it proves a native tonic client
/// — which is what the checkpoint-blob-indexer uses — works end-to-end against soma's
/// rpc server (Path A: no dedicated tonic Server needed). Also checks that a fetched
/// checkpoint survives the encode→decode round-trip used by the object-store path.
#[cfg(msim)]
#[msim::sim_test]
async fn test_rpc_ingestion_source() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let rpc_url = test_cluster.fullnode_handle.rpc_url.clone();
    info!("Fullnode gRPC endpoint: {}", rpc_url);

    // Build an IngestionClient backed by the gRPC unary source.
    let registry = prometheus::Registry::new();
    let metrics = IngestionMetrics::new(None, &registry);
    let args = IngestionClientArgs {
        rpc_api_url: Some(rpc_url.parse().expect("rpc_url should be a valid URL")),
        ..Default::default()
    };
    let client = IngestionClient::new(args, metrics).expect("build gRPC ingestion client");

    // Genesis checkpoint must be fetchable.
    let genesis: Arc<Checkpoint> = client.fetch(0).await.expect("fetch checkpoint 0 over gRPC");
    assert_eq!(genesis.summary.sequence_number, 0);
    info!(
        "Fetched genesis checkpoint over gRPC: {} transactions",
        genesis.transactions.len()
    );

    // A real fetched checkpoint survives the .binpb.zst encode→decode round-trip.
    let encoded = checkpoint_blob::encode_checkpoint(&genesis).expect("encode checkpoint");
    let decoded = checkpoint_blob::decode_checkpoint(&encoded).expect("decode checkpoint");
    assert_eq!(decoded.summary.sequence_number, genesis.summary.sequence_number);
    assert_eq!(decoded.summary.epoch, genesis.summary.epoch);
    assert_eq!(decoded.transactions.len(), genesis.transactions.len());

    // Execute a transaction, then fetch the checkpoints that follow over gRPC.
    let addresses = test_cluster.wallet.get_addresses();
    let tx_data = e2e_tests::balance_transfer_data(
        &test_cluster,
        types::object::CoinType::Usdc,
        addresses[0],
        vec![(addresses[1], 1000)],
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    let mut fetched_seq = 0;
    let start = tokio::time::Instant::now();
    while start.elapsed() < Duration::from_secs(60) {
        match client.fetch(fetched_seq + 1).await {
            Ok(cp) => {
                assert_eq!(cp.summary.sequence_number, fetched_seq + 1);
                fetched_seq += 1;
                if fetched_seq >= 2 {
                    break;
                }
            }
            Err(_) => sleep(Duration::from_millis(200)).await,
        }
    }
    assert!(
        fetched_seq >= 2,
        "expected to fetch at least 2 non-genesis checkpoints over gRPC, got {fetched_seq}",
    );
    info!("Fetched checkpoints 0..={} over the gRPC ingestion source", fetched_seq);
}

/// Verify the gRPC streaming ingestion source (StreamingIngestionClient, exposed via the
/// `--streaming-url` flag). The client runs a background subscribe_checkpoints subscription
/// feeding a bounded buffer, and falls back to unary get_checkpoint for backfill. Fetching a
/// contiguous run of checkpoints exercises both: early ones (genesis) are backfilled via RPC,
/// later ones — produced by the chain while the test runs — arrive over the live stream.
#[cfg(msim)]
#[msim::sim_test]
async fn test_streaming_ingestion_source() {
    init_tracing();

    let test_cluster = TestClusterBuilder::new().build().await;
    let rpc_url = test_cluster.fullnode_handle.rpc_url.clone();
    info!("Fullnode gRPC endpoint: {}", rpc_url);

    let registry = prometheus::Registry::new();
    let metrics = IngestionMetrics::new(None, &registry);
    let args = IngestionClientArgs {
        streaming_url: Some(rpc_url.parse().expect("rpc_url should be a valid URL")),
        ..Default::default()
    };
    let client = IngestionClient::new(args, metrics).expect("build streaming ingestion client");

    // Drive checkpoint production so the live stream has something to deliver.
    let addresses = test_cluster.wallet.get_addresses();
    let tx_data = e2e_tests::balance_transfer_data(
        &test_cluster,
        types::object::CoinType::Usdc,
        addresses[0],
        vec![(addresses[1], 1000)],
    );
    let response = test_cluster.sign_and_execute_transaction(&tx_data).await;
    assert!(response.effects.status().is_ok());

    // Fetch a contiguous run 0..=4. Genesis is a backfill (RPC) fetch; later checkpoints are
    // produced while the test runs and must resolve via the subscription. Each fetch is
    // bounded by wait_for's retry loop; the whole run is capped so a wedged stream fails the
    // test instead of hanging it.
    for seq in 0..=4u64 {
        let checkpoint: Arc<Checkpoint> = tokio::time::timeout(
            Duration::from_secs(120),
            client.wait_for(seq, Duration::from_millis(200)),
        )
        .await
        .unwrap_or_else(|_| panic!("streaming fetch of checkpoint {seq} timed out"))
        .unwrap_or_else(|e| panic!("streaming fetch of checkpoint {seq} failed: {e}"));
        assert_eq!(checkpoint.summary.sequence_number, seq);
        info!(
            "Streaming source delivered checkpoint {} ({} transactions)",
            seq,
            checkpoint.transactions.len()
        );
    }

    info!("Streaming ingestion source delivered checkpoints 0..=4");
}
