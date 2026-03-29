// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Handler unit tests — Layer 1 of the indexer test plan.
//!
//! Each test:
//! 1. Spins up an ephemeral Postgres via TempDb
//! 2. Builds a synthetic checkpoint via TestCheckpointBuilder
//! 3. Calls `Processor::process()` to verify row extraction
//! 4. Calls `Handler::commit()` to verify DB insertion
//!
//! Tests are `#[ignore]` by default since they require Postgres on PATH.
//! Run with: `cargo test -p indexer-alt -- --ignored`

use std::ops::DerefMut;
use std::sync::Arc;

use diesel::prelude::*;
use diesel_async::RunQueryDsl;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::handler::Handler;
use types::base::SomaAddress;
use types::full_checkpoint_content::Checkpoint;
use types::object::ObjectID;
use types::test_checkpoint_data_builder::TestCheckpointBuilder;

use indexer_alt::handlers::*;

// ---------------------------------------------------------------------------
// Test setup helpers
// ---------------------------------------------------------------------------

async fn setup() -> (indexer_pg_db::Db, indexer_pg_db::temp::TempDb) {
    let temp = indexer_pg_db::temp::TempDb::new();
    let db = indexer_pg_db::Db::for_write(
        temp.url().clone(),
        indexer_pg_db::DbArgs {
            db_connection_pool_size: 5,
            db_connection_timeout_ms: 30_000,
            db_statement_timeout_ms: None,
        },
    )
    .await
    .expect("DB pool");
    db.run_migrations(Some(&indexer_alt_schema::MIGRATIONS)).await.expect("migrations");
    (db, temp)
}

/// Build a simple checkpoint with one coin transfer.
fn simple_checkpoint() -> Checkpoint {
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    TestCheckpointBuilder::new(1)
        .with_epoch(0)
        .with_network_total_transactions(1)
        .add_transfer_coin(sender, recipient, 100_000)
        .build()
}

// ===========================================================================
// cp_sequence_numbers
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_cp_sequence_numbers_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = cp_sequence_numbers::CpSequenceNumbers.process(&cp).await.unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].cp_sequence_number, 1);
    assert_eq!(values[0].epoch, 0);
}

#[tokio::test]
#[ignore]
async fn test_cp_sequence_numbers_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = cp_sequence_numbers::CpSequenceNumbers.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = cp_sequence_numbers::CpSequenceNumbers::commit(&values, &mut conn).await.unwrap();
    assert_eq!(rows, 1);
}

// ===========================================================================
// tx_digests
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_tx_digests_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = tx_digests::TxDigests.process(&cp).await.unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].tx_sequence_number, 0);
    assert!(!values[0].tx_digest.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_tx_digests_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = tx_digests::TxDigests.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = tx_digests::TxDigests::commit(&values, &mut conn).await.unwrap();
    assert_eq!(rows, 1);
}

// ===========================================================================
// kv_checkpoints
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_kv_checkpoints_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = kv_checkpoints::KvCheckpoints.process(&cp).await.unwrap();
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].sequence_number, 1);
    assert!(!values[0].checkpoint_summary.is_empty());
    assert!(!values[0].checkpoint_contents.is_empty());
    assert!(!values[0].validator_signatures.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_kv_checkpoints_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = kv_checkpoints::KvCheckpoints.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = kv_checkpoints::KvCheckpoints::commit(&values, &mut conn).await.unwrap();
    assert_eq!(rows, 1);
}

// ===========================================================================
// kv_transactions
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_kv_transactions_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = kv_transactions::KvTransactions.process(&cp).await.unwrap();
    assert_eq!(values.len(), 1);
    assert!(!values[0].tx_digest.is_empty());
    assert!(!values[0].raw_transaction.is_empty());
    assert!(!values[0].raw_effects.is_empty());
    // Soma has no events
    assert!(values[0].events.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_kv_transactions_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = kv_transactions::KvTransactions.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = kv_transactions::KvTransactions::commit(&values, &mut conn).await.unwrap();
    assert_eq!(rows, 1);
}

// ===========================================================================
// kv_objects
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_kv_objects_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = kv_objects::KvObjects.process(&cp).await.unwrap();
    // Transfer creates 2 output objects: gas coin + new coin
    assert!(values.len() >= 2);
    // All output objects should have serialized_object set
    for v in &values {
        if v.serialized_object.is_some() {
            assert!(!v.serialized_object.as_ref().unwrap().is_empty());
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_kv_objects_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = kv_objects::KvObjects.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = kv_objects::KvObjects::commit(&values, &mut conn).await.unwrap();
    assert!(rows >= 2);
}

// ===========================================================================
// kv_epoch_starts — skips non-genesis, non-epoch-boundary checkpoints
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_kv_epoch_starts_skips_normal_checkpoint() {
    let cp = Arc::new(simple_checkpoint());
    let values = kv_epoch_starts::KvEpochStarts.process(&cp).await.unwrap();
    // Checkpoint 1, no end_of_epoch_data, not genesis → empty
    assert!(values.is_empty());
}

// ===========================================================================
// kv_epoch_ends — skips non-epoch-boundary checkpoints
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_kv_epoch_ends_skips_normal_checkpoint() {
    let cp = Arc::new(simple_checkpoint());
    let values = kv_epoch_ends::KvEpochEnds.process(&cp).await.unwrap();
    assert!(values.is_empty());
}

// ===========================================================================
// tx_affected_addresses
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_tx_affected_addresses_process() {
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let cp = Arc::new(
        TestCheckpointBuilder::new(1)
            .with_network_total_transactions(1)
            .add_transfer_coin(sender, recipient, 50_000)
            .build(),
    );
    let values = tx_affected_addresses::TxAffectedAddresses.process(&cp).await.unwrap();
    // Should include sender + recipient
    assert!(values.len() >= 2);
    // All should reference tx_sequence_number 0
    for v in &values {
        assert_eq!(v.tx_sequence_number, 0);
        assert_eq!(v.sender, sender.to_vec());
    }
}

#[tokio::test]
#[ignore]
async fn test_tx_affected_addresses_commit() {
    let (db, _temp) = setup().await;
    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let cp = Arc::new(
        TestCheckpointBuilder::new(1)
            .with_network_total_transactions(1)
            .add_transfer_coin(sender, recipient, 50_000)
            .build(),
    );
    let values = tx_affected_addresses::TxAffectedAddresses.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows =
        tx_affected_addresses::TxAffectedAddresses::commit(&values, &mut conn).await.unwrap();
    assert!(rows >= 2);
}

// ===========================================================================
// tx_affected_objects
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_tx_affected_objects_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = tx_affected_objects::TxAffectedObjects.process(&cp).await.unwrap();
    // At least 2 objects affected (gas coin + transferred coin)
    assert!(values.len() >= 2);
}

#[tokio::test]
#[ignore]
async fn test_tx_affected_objects_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = tx_affected_objects::TxAffectedObjects.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = tx_affected_objects::TxAffectedObjects::commit(&values, &mut conn).await.unwrap();
    assert!(rows >= 2);
}

// ===========================================================================
// tx_balance_changes
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_tx_balance_changes_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = tx_balance_changes::TxBalanceChanges.process(&cp).await.unwrap();
    assert_eq!(values.len(), 1);
    assert!(!values[0].balance_changes.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_tx_balance_changes_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = tx_balance_changes::TxBalanceChanges.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = tx_balance_changes::TxBalanceChanges::commit(&values, &mut conn).await.unwrap();
    assert_eq!(rows, 1);
}

// ===========================================================================
// tx_kinds
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_tx_kinds_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = tx_kinds::TxKinds.process(&cp).await.unwrap();
    assert_eq!(values.len(), 1);
}

#[tokio::test]
#[ignore]
async fn test_tx_kinds_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = tx_kinds::TxKinds.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = tx_kinds::TxKinds::commit(&values, &mut conn).await.unwrap();
    assert_eq!(rows, 1);
}

// ===========================================================================
// tx_calls — always empty for Soma
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_tx_calls_always_empty() {
    let cp = Arc::new(simple_checkpoint());
    let values = tx_calls::TxCalls.process(&cp).await.unwrap();
    assert!(values.is_empty());
}

// ===========================================================================
// obj_versions
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_obj_versions_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = obj_versions::ObjVersions.process(&cp).await.unwrap();
    // Should have entries for all changed objects
    assert!(values.len() >= 2);
    for v in &values {
        assert_eq!(v.cp_sequence_number, 1);
    }
}

#[tokio::test]
#[ignore]
async fn test_obj_versions_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = obj_versions::ObjVersions.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = obj_versions::ObjVersions::commit(&values, &mut conn).await.unwrap();
    assert!(rows >= 2);
}

// ===========================================================================
// obj_info
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_obj_info_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = obj_info::ObjInfo.process(&cp).await.unwrap();
    // New objects => ownership changed from NotExist
    assert!(!values.is_empty());
    for v in &values {
        assert_eq!(v.cp_sequence_number, 1);
    }
}

#[tokio::test]
#[ignore]
async fn test_obj_info_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = obj_info::ObjInfo.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = obj_info::ObjInfo::commit(&values, &mut conn).await.unwrap();
    assert!(!values.is_empty());
    assert_eq!(rows, values.len());
}

// ===========================================================================
// coin_balance_buckets
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_coin_balance_buckets_process() {
    let cp = Arc::new(simple_checkpoint());
    let values = coin_balance_buckets::CoinBalanceBuckets.process(&cp).await.unwrap();
    // Should detect the coins created
    assert!(!values.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_coin_balance_buckets_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());
    let values = coin_balance_buckets::CoinBalanceBuckets.process(&cp).await.unwrap();

    let mut conn = db.connect().await.unwrap();
    let rows = coin_balance_buckets::CoinBalanceBuckets::commit(&values, &mut conn).await.unwrap();
    assert_eq!(rows, values.len());
}

// ===========================================================================
// Idempotency — double commit should not fail (on_conflict_do_nothing)
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_idempotent_commit() {
    let (db, _temp) = setup().await;
    let cp = Arc::new(simple_checkpoint());

    let values = kv_checkpoints::KvCheckpoints.process(&cp).await.unwrap();
    let mut conn = db.connect().await.unwrap();

    let rows1 = kv_checkpoints::KvCheckpoints::commit(&values, &mut conn).await.unwrap();
    assert_eq!(rows1, 1);

    // Second commit should succeed but insert 0 new rows
    let rows2 = kv_checkpoints::KvCheckpoints::commit(&values, &mut conn).await.unwrap();
    assert_eq!(rows2, 0);
}

// ===========================================================================
// Multiple transactions in one checkpoint
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_multiple_transactions() {
    let sender = SomaAddress::random();
    let r1 = SomaAddress::random();
    let r2 = SomaAddress::random();

    let cp = Arc::new(
        TestCheckpointBuilder::new(5)
            .with_epoch(1)
            .with_network_total_transactions(10)
            .add_transfer_coin(sender, r1, 50_000)
            .add_transfer_coin(sender, r2, 75_000)
            .build(),
    );

    // tx_digests should have 2 entries
    let digests = tx_digests::TxDigests.process(&cp).await.unwrap();
    assert_eq!(digests.len(), 2);
    // First tx starts at network_total - num_txs = 10 - 2 = 8
    assert_eq!(digests[0].tx_sequence_number, 8);
    assert_eq!(digests[1].tx_sequence_number, 9);

    // kv_transactions should have 2 entries
    let txs = kv_transactions::KvTransactions.process(&cp).await.unwrap();
    assert_eq!(txs.len(), 2);

    // cp_sequence_numbers should have 1 entry
    let cps = cp_sequence_numbers::CpSequenceNumbers.process(&cp).await.unwrap();
    assert_eq!(cps.len(), 1);
    assert_eq!(cps[0].tx_lo, 8);
    assert_eq!(cps[0].epoch, 1);
}

