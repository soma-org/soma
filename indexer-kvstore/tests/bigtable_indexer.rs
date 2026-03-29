// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Full-stack BigTable indexer e2e test: write `.binpb.zst` checkpoint files
//! to disk, run the BigTableIndexer pipeline engine, then verify data in BigTable.
//!
//! Requires the BigTable emulator (`cbtemulator` + `cbt`).
//! Run with:
//!   PYO3_PYTHON=python3 cargo test -p indexer-kvstore --test bigtable_indexer -- --ignored

mod emulator;

use std::time::Duration;

use indexer_framework::IndexerArgs;
use indexer_framework::ingestion::ClientArgs;
use indexer_framework::ingestion::ingestion_client::IngestionClientArgs;
use indexer_framework::pipeline::CommitterConfig;
use indexer_kvstore::{
    BigTableClient, BigTableIndexer, BigTableStore, IndexerConfig, KeyValueStoreReader,
    PipelineLayer,
};
use rpc::utils::checkpoint_blob;
use types::base::SomaAddress;
use types::test_checkpoint_data_builder::{
    TestCheckpointBuilder, default_test_system_state,
};

/// Write a checkpoint to disk as `{seq}.binpb.zst`.
fn write_checkpoint_file(
    dir: &std::path::Path,
    checkpoint: &types::full_checkpoint_content::Checkpoint,
) {
    let bytes =
        checkpoint_blob::encode_checkpoint(checkpoint).expect("Failed to encode checkpoint");
    let path = dir.join(format!("{}.binpb.zst", checkpoint.summary.sequence_number));
    std::fs::write(path, bytes).expect("Failed to write checkpoint file");
}

/// Poll BigTable watermarks until they reach the expected checkpoint, or timeout.
async fn wait_for_watermark(
    client: &mut BigTableClient,
    expected_checkpoint: u64,
    timeout_duration: Duration,
) {
    let start = std::time::Instant::now();
    loop {
        if start.elapsed() > timeout_duration {
            panic!("Timed out waiting for watermark to reach checkpoint {}", expected_checkpoint);
        }
        if let Ok(Some(wm)) = client.get_watermark().await {
            if wm.checkpoint_hi_inclusive >= expected_checkpoint {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ---------- Full-stack indexer test ----------

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_bigtable_indexer_e2e() {
    let checkpoint_dir = tempfile::tempdir().unwrap();
    let dir = checkpoint_dir.path();

    // 1. Build synthetic checkpoints
    let cp0 = TestCheckpointBuilder::new(0)
        .with_genesis_system_state(default_test_system_state())
        .build();

    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let cp1 = TestCheckpointBuilder::new(1).add_transfer_coin(sender, recipient, 5000).build();

    // 2. Write checkpoint files to disk
    write_checkpoint_file(dir, &cp0);
    write_checkpoint_file(dir, &cp1);

    // 3. Start BigTable emulator and create tables
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();

    // 4. Create BigTable store
    let client = emulator::client(emu.host()).await.unwrap();
    let store = BigTableStore::new(client);

    // 5. Create and run indexer
    let registry = prometheus::Registry::new();
    let client_args = ClientArgs {
        ingestion: IngestionClientArgs {
            local_ingestion_path: Some(dir.to_path_buf()),
            ..Default::default()
        },
    };

    let committer = CommitterConfig {
        watermark_interval_ms: 100,
        collect_interval_ms: 100,
        ..Default::default()
    };

    let bigtable_indexer = BigTableIndexer::new(
        store,
        IndexerArgs { last_checkpoint: Some(1), ..Default::default() },
        client_args,
        Default::default(),
        committer,
        IndexerConfig::default(),
        PipelineLayer::default(),
        &registry,
    )
    .await
    .unwrap();

    // Run the indexer in a background task
    let mut service = bigtable_indexer.indexer.run().await.unwrap();
    let indexer_handle = tokio::spawn(async move {
        let _ = service.join().await;
    });

    // 6. Wait for watermarks to advance
    let mut reader = emulator::client(emu.host()).await.unwrap();
    wait_for_watermark(&mut reader, 1, Duration::from_secs(60)).await;

    // 7. Verify checkpoint data
    let checkpoints = reader.get_checkpoints(&[0, 1]).await.unwrap();
    assert_eq!(checkpoints.len(), 2, "should have 2 checkpoints");
    assert_eq!(checkpoints[0].summary.sequence_number, 0);
    assert_eq!(checkpoints[1].summary.sequence_number, 1);

    // 8. Verify checkpoint-by-digest lookup
    let by_digest = reader
        .get_checkpoint_by_digest(*cp0.summary.digest())
        .await
        .unwrap()
        .expect("checkpoint 0 should be findable by digest");
    assert_eq!(by_digest.summary.sequence_number, 0);

    // 9. Verify transaction data
    let tx_digest = cp1.contents.iter().next().unwrap().transaction;
    let txs = reader.get_transactions(&[tx_digest]).await.unwrap();
    assert_eq!(txs.len(), 1, "should have 1 transaction");
    assert_eq!(txs[0].checkpoint_number, 1);

    // 10. Verify object data
    let first_tx = &cp1.transactions[0];
    let output_objects: Vec<_> = first_tx.output_objects(&cp1.object_set).collect();
    assert!(!output_objects.is_empty(), "should have output objects");

    let obj = output_objects[0];
    let key = types::storage::ObjectKey(obj.id(), obj.version());
    let objects = reader.get_objects(&[key]).await.unwrap();
    assert_eq!(objects.len(), 1, "should have 1 object");
    assert_eq!(objects[0].id(), obj.id());

    // 11. Verify epoch data
    let epoch = reader.get_epoch(0).await.unwrap().expect("epoch 0 should exist");
    assert_eq!(epoch.epoch, Some(0));
    assert!(epoch.system_state_bcs.is_some(), "should have system state");

    // 12. Verify watermarks are consistent
    let wm = reader.get_watermark().await.unwrap().expect("watermark should exist");
    assert_eq!(wm.checkpoint_hi_inclusive, 1);

    // Clean up: abort the indexer task (it should have finished, but just in case)
    indexer_handle.abort();
    let _ = indexer_handle.await;
}
