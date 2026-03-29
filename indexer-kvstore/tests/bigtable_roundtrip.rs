// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! BigTable roundtrip tests: process → write → read → verify.
//!
//! All tests are `#[ignore]` and require the BigTable emulator (`cbtemulator` + `cbt`).
//! Run with:
//!   PYO3_PYTHON=python3 cargo test -p indexer-kvstore --test bigtable_roundtrip -- --ignored

mod emulator;

use std::sync::Arc;

use indexer_framework::pipeline::Processor;
use indexer_kvstore::{
    BigTableClient, CheckpointsByDigestPipeline, CheckpointsPipeline, EpochEndPipeline,
    EpochStartPipeline, KeyValueStoreReader, ObjectsPipeline,
    TransactionsPipeline, Watermark,
};
use types::base::SomaAddress;
use types::committee::Committee;
use types::full_checkpoint_content::Checkpoint;
use types::test_checkpoint_data_builder::{
    TestCheckpointBuilder, default_test_system_state,
};

/// Set watermarks for all pipelines to the given values.
async fn set_watermarks(client: &mut BigTableClient, wm: &Watermark) {
    for name in indexer_kvstore::ALL_PIPELINE_NAMES {
        client.set_pipeline_watermark(name, wm).await.unwrap();
    }
}

/// Build a genesis checkpoint with system state.
fn genesis_checkpoint() -> Checkpoint {
    TestCheckpointBuilder::new(0)
        .with_genesis_system_state(default_test_system_state())
        .build()
}

/// Build a checkpoint at sequence_number=1 with a transfer.
fn transfer_checkpoint() -> Checkpoint {
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    TestCheckpointBuilder::new(1).add_transfer_coin(sender, recipient, 1000).build()
}

/// Helper to start emulator and create tables.
async fn setup() -> (emulator::BigTableEmulator, BigTableClient) {
    let emu = emulator::BigTableEmulator::start().unwrap();
    emulator::create_tables(emu.host(), emulator::INSTANCE_ID).await.unwrap();
    let client = emulator::client(emu.host()).await.unwrap();
    (emu, client)
}

// ---------- Checkpoint roundtrip ----------

#[tokio::test]
#[ignore]
async fn test_checkpoint_roundtrip() {
    let (_emu, mut client) = setup().await;

    let cp = genesis_checkpoint();
    let entries = CheckpointsPipeline.process(&Arc::new(cp.clone())).await.unwrap();
    client.write_entries(indexer_kvstore::tables::checkpoints::NAME, entries).await.unwrap();

    let mut results = client.get_checkpoints(&[0]).await.unwrap();
    assert_eq!(results.len(), 1);
    let data = results.pop().unwrap();
    assert_eq!(data.summary.sequence_number, 0);
}

// ---------- Checkpoint by digest roundtrip ----------

#[tokio::test]
#[ignore]
async fn test_checkpoint_by_digest_roundtrip() {
    let (_emu, mut client) = setup().await;

    let cp = genesis_checkpoint();
    let arc = Arc::new(cp.clone());

    let entries = CheckpointsPipeline.process(&arc).await.unwrap();
    client.write_entries(indexer_kvstore::tables::checkpoints::NAME, entries).await.unwrap();

    let entries = CheckpointsByDigestPipeline.process(&arc).await.unwrap();
    client
        .write_entries(indexer_kvstore::tables::checkpoints_by_digest::NAME, entries)
        .await
        .unwrap();

    let data = client
        .get_checkpoint_by_digest(*cp.summary.digest())
        .await
        .unwrap()
        .expect("checkpoint not found by digest");
    assert_eq!(data.summary.sequence_number, 0);
}

// ---------- Transaction roundtrip ----------

#[tokio::test]
#[ignore]
async fn test_transaction_roundtrip() {
    let (_emu, mut client) = setup().await;

    let cp = transfer_checkpoint();
    let entries = TransactionsPipeline.process(&Arc::new(cp.clone())).await.unwrap();
    client.write_entries(indexer_kvstore::tables::transactions::NAME, entries).await.unwrap();

    let tx_digest = cp.contents.iter().next().unwrap().transaction;
    let results = client.get_transactions(&[tx_digest]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].checkpoint_number, 1);
}

// ---------- Object roundtrip ----------

#[tokio::test]
#[ignore]
async fn test_object_roundtrip() {
    let (_emu, mut client) = setup().await;

    let cp = transfer_checkpoint();
    let entries = ObjectsPipeline.process(&Arc::new(cp.clone())).await.unwrap();
    client.write_entries(indexer_kvstore::tables::objects::NAME, entries).await.unwrap();

    let first_tx = &cp.transactions[0];
    let output_objects: Vec<_> = first_tx.output_objects(&cp.object_set).collect();
    assert!(!output_objects.is_empty(), "should have output objects");

    let obj = output_objects[0];
    let key = types::storage::ObjectKey(obj.id(), obj.version());
    let results = client.get_objects(&[key]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id(), obj.id());
}

// ---------- Latest object roundtrip ----------

#[tokio::test]
#[ignore]
async fn test_latest_object_roundtrip() {
    let (_emu, mut client) = setup().await;

    let cp = transfer_checkpoint();
    let entries = ObjectsPipeline.process(&Arc::new(cp.clone())).await.unwrap();
    client.write_entries(indexer_kvstore::tables::objects::NAME, entries).await.unwrap();

    let first_tx = &cp.transactions[0];
    let output_objects: Vec<_> = first_tx.output_objects(&cp.object_set).collect();
    let obj = output_objects[0];

    let latest =
        client.get_latest_object(&obj.id()).await.unwrap().expect("latest object not found");
    assert_eq!(latest.id(), obj.id());
}

// ---------- Epoch roundtrip ----------

#[tokio::test]
#[ignore]
async fn test_epoch_roundtrip() {
    let (_emu, mut client) = setup().await;

    let cp = genesis_checkpoint();
    let entries = EpochStartPipeline.process(&Arc::new(cp)).await.unwrap();
    client.write_entries(indexer_kvstore::tables::epochs::NAME, entries).await.unwrap();

    let epoch_data = client.get_epoch(0).await.unwrap().expect("epoch 0 not found");
    assert_eq!(epoch_data.epoch, Some(0));
    assert!(epoch_data.system_state_bcs.is_some());
}

// ---------- Epoch end roundtrip ----------

#[tokio::test]
#[ignore]
async fn test_epoch_end_roundtrip() {
    let (_emu, mut client) = setup().await;

    let (next_committee, _) = Committee::new_simple_test_committee_of_size(4);
    let system_state = default_test_system_state();
    let cp = TestCheckpointBuilder::new(10)
        .with_epoch(0)
        .with_end_of_epoch(next_committee)
        .add_change_epoch(system_state)
        .build();

    let entries = EpochEndPipeline.process(&Arc::new(cp)).await.unwrap();
    client.write_entries(indexer_kvstore::tables::epochs::NAME, entries).await.unwrap();

    let epoch_data = client.get_epoch(0).await.unwrap().expect("epoch 0 not found after end");
    assert!(epoch_data.end_checkpoint.is_some());
}

// ---------- Latest epoch roundtrip ----------

#[tokio::test]
#[ignore]
async fn test_latest_epoch_roundtrip() {
    let (_emu, mut client) = setup().await;

    let cp = genesis_checkpoint();
    let entries = EpochStartPipeline.process(&Arc::new(cp)).await.unwrap();
    client.write_entries(indexer_kvstore::tables::epochs::NAME, entries).await.unwrap();

    let latest = client.get_latest_epoch().await.unwrap().expect("should find latest epoch");
    assert_eq!(latest.epoch, Some(0));
}

// ---------- Watermark roundtrip ----------

#[tokio::test]
#[ignore]
async fn test_watermark_roundtrip() {
    let (_emu, mut client) = setup().await;

    let wm = Watermark {
        epoch_hi_inclusive: 5,
        checkpoint_hi_inclusive: 100,
        tx_hi: 500,
        timestamp_ms_hi_inclusive: 1234567890,
    };
    set_watermarks(&mut client, &wm).await;

    let read_wm = client.get_watermark().await.unwrap().expect("watermark should exist");
    assert_eq!(read_wm.epoch_hi_inclusive, 5);
    assert_eq!(read_wm.checkpoint_hi_inclusive, 100);
    assert_eq!(read_wm.tx_hi, 500);
    assert_eq!(read_wm.timestamp_ms_hi_inclusive, 1234567890);
}

// ---------- Multi-checkpoint batch ----------

#[tokio::test]
#[ignore]
async fn test_multi_checkpoint_batch() {
    let (_emu, mut client) = setup().await;

    for seq in 0..3u64 {
        let cp = TestCheckpointBuilder::new(seq).build();
        let entries = CheckpointsPipeline.process(&Arc::new(cp)).await.unwrap();
        client.write_entries(indexer_kvstore::tables::checkpoints::NAME, entries).await.unwrap();
    }

    let results = client.get_checkpoints(&[0, 1, 2]).await.unwrap();
    assert_eq!(results.len(), 3, "should read back all 3 checkpoints");
    for (i, data) in results.iter().enumerate() {
        assert_eq!(data.summary.sequence_number, i as u64);
    }
}
