// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests — Layer 4 of the indexer test plan.
//!
//! These tests verify the full data flow:
//!   TestCheckpointBuilder → Handler::process() → Handler::commit() → GraphQL query
//!
//! This validates that data written by indexer handlers is correctly queryable
//! through the GraphQL API, testing the entire indexer→DB→GraphQL stack.
//!
//! Tests are `#[ignore]` by default since they require Postgres on PATH.
//! Run with: `cargo test -p soma-graphql --test e2e_tests -- --ignored`

use std::ops::DerefMut;
use std::sync::Arc;

use diesel_async::RunQueryDsl;
use serde_json::Value;

use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::handler::Handler;
use indexer_pg_db::DbArgs;
use types::base::SomaAddress;
use types::full_checkpoint_content::Checkpoint;
use types::object::ObjectID;
use types::target::TargetStatus;
use types::test_checkpoint_data_builder::{test_filled_target, test_target, TestCheckpointBuilder};

use indexer_alt::handlers::*;
use soma_graphql::config::GraphQlConfig;
use soma_graphql::db::PgReader;
use soma_graphql::{build_schema, SomaSchema};

// ---------------------------------------------------------------------------
// Test setup helpers
// ---------------------------------------------------------------------------

struct TestContext {
    schema: SomaSchema,
    db: indexer_pg_db::Db,
    _temp: indexer_pg_db::temp::TempDb,
}

async fn setup() -> TestContext {
    let temp = indexer_pg_db::temp::TempDb::new();

    let db = indexer_pg_db::Db::for_write(
        temp.url().clone(),
        DbArgs {
            db_connection_pool_size: 5,
            db_connection_timeout_ms: 30_000,
            db_statement_timeout_ms: None,
        },
    )
    .await
    .expect("DB pool");

    db.run_migrations(Some(&indexer_alt_schema::MIGRATIONS))
        .await
        .expect("migrations");

    let pg = Arc::new(
        PgReader::new(
            temp.url().clone(),
            DbArgs {
                db_connection_pool_size: 5,
                db_connection_timeout_ms: 30_000,
                db_statement_timeout_ms: None,
            },
        )
        .await
        .expect("PgReader"),
    );

    let config = GraphQlConfig::default();
    let schema = build_schema(pg, config);

    TestContext {
        schema,
        db,
        _temp: temp,
    }
}

async fn execute(schema: &SomaSchema, query: &str) -> Value {
    let resp = schema.execute(query).await;
    let json = serde_json::to_value(&resp).unwrap();
    assert!(
        resp.errors.is_empty(),
        "GraphQL errors: {:?}",
        resp.errors
    );
    json
}

/// Process a checkpoint through a handler and commit the results to the database.
async fn process_and_commit<H: Handler + Processor>(
    handler: &H,
    checkpoint: &Arc<Checkpoint>,
    db: &indexer_pg_db::Db,
) -> Vec<H::Value> {
    let values = handler.process(checkpoint).await.expect("process");
    if !values.is_empty() {
        let mut conn = db.connect().await.unwrap();
        H::commit(&values, &mut conn).await.expect("commit");
    }
    values
}

// ---------------------------------------------------------------------------
// E2e: Transfer coin → kv_transactions + tx_digests + tx_balance_changes → GraphQL
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_e2e_transfer_coin_to_graphql_transaction() {
    let ctx = setup().await;

    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();
    let checkpoint = Arc::new(
        TestCheckpointBuilder::new(0)
            .add_transfer_coin(sender, recipient, 5000)
            .build(),
    );

    // Process and commit through indexer handlers
    process_and_commit(&kv_transactions::KvTransactions, &checkpoint, &ctx.db).await;
    process_and_commit(&tx_digests::TxDigests, &checkpoint, &ctx.db).await;
    process_and_commit(&kv_checkpoints::KvCheckpoints, &checkpoint, &ctx.db).await;
    process_and_commit(
        &cp_sequence_numbers::CpSequenceNumbers,
        &checkpoint,
        &ctx.db,
    )
    .await;

    // Verify via GraphQL: fetch the transaction by its digest
    let tx = &checkpoint.transactions[0];
    let digest_b58 = bs58::encode(tx.transaction.digest().inner()).into_string();

    let query = format!(
        r#"{{ transaction(digest: "{}") {{ checkpointSequenceNumber }} }}"#,
        digest_b58
    );
    let json = execute(&ctx.schema, &query).await;
    assert_eq!(
        json["data"]["transaction"]["checkpointSequenceNumber"],
        "0"
    );

    // Verify checkpoint query works too
    let json = execute(
        &ctx.schema,
        r#"{ checkpoint(sequenceNumber: 0) { sequenceNumber epoch } }"#,
    )
    .await;
    assert_eq!(json["data"]["checkpoint"]["sequenceNumber"], "0");
    assert_eq!(json["data"]["checkpoint"]["epoch"], "0");
}

// ---------------------------------------------------------------------------
// E2e: Target indexing → soma_targets → GraphQL target query
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_e2e_target_indexing_to_graphql() {
    let ctx = setup().await;

    let target = test_target(1, TargetStatus::Open, 10_000);
    let checkpoint = Arc::new(TestCheckpointBuilder::new(5).with_epoch(1).add_target(target).build());

    // Process through soma_targets handler
    let values = process_and_commit(&soma_targets::SomaTargets, &checkpoint, &ctx.db).await;
    assert_eq!(values.len(), 1);

    // Extract the target_id from the handler output
    let target_id_hex = format!("0x{}", hex::encode(&values[0].target_id));

    // Query via GraphQL
    let query = format!(
        r#"{{ target(targetId: "{}") {{ status epoch rewardPool }} }}"#,
        target_id_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let t = &json["data"]["target"];
    assert_eq!(t["status"], "open");
    assert_eq!(t["epoch"], "1");
    assert_eq!(t["rewardPool"], "10000");
}

// ---------------------------------------------------------------------------
// E2e: Multiple targets → paginated targets query with filter
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_e2e_targets_pagination_and_filter() {
    let ctx = setup().await;

    // Create checkpoint with multiple targets (open and filled)
    let open_target = test_target(0, TargetStatus::Open, 5000);
    let model_id = ObjectID::random();
    let submitter = SomaAddress::random();
    let filled_target = test_filled_target(0, 0, submitter, model_id, 8000, 1000);

    let checkpoint = Arc::new(
        TestCheckpointBuilder::new(10)
            .add_target(open_target)
            .add_target(filled_target)
            .build(),
    );

    let values = process_and_commit(&soma_targets::SomaTargets, &checkpoint, &ctx.db).await;
    assert_eq!(values.len(), 2);

    // Query all targets
    let json = execute(
        &ctx.schema,
        r#"{ targets { edges { node { status rewardPool } } } }"#,
    )
    .await;
    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);

    // Filter by status=open (handler stores lowercase)
    let json = execute(
        &ctx.schema,
        r#"{ targets(filter: { status: "open" }) { edges { node { status } } } }"#,
    )
    .await;
    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["node"]["status"], "open");
}

// ---------------------------------------------------------------------------
// E2e: ClaimRewards → soma_rewards → GraphQL rewards query
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_e2e_rewards_to_graphql() {
    let ctx = setup().await;

    let sender = SomaAddress::random();
    let target_id = ObjectID::random();
    let checkpoint = Arc::new(
        TestCheckpointBuilder::new(3)
            .with_epoch(2)
            .add_claim_rewards(sender, target_id, 500)
            .build(),
    );

    let values =
        process_and_commit(&soma_rewards::SomaRewards, &checkpoint, &ctx.db).await;
    assert_eq!(values.len(), 1);

    // Query via GraphQL
    let json = execute(
        &ctx.schema,
        r#"{ rewards(epoch: 2) { epoch txDigest } }"#,
    )
    .await;

    let rewards = json["data"]["rewards"].as_array().unwrap();
    assert_eq!(rewards.len(), 1);
    assert_eq!(rewards[0]["epoch"], "2");
}

// ---------------------------------------------------------------------------
// E2e: Address transactions flow
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_e2e_address_transactions() {
    let ctx = setup().await;

    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();

    let checkpoint = Arc::new(
        TestCheckpointBuilder::new(0)
            .add_transfer_coin(sender, recipient, 1000)
            .add_transfer_coin(sender, recipient, 2000)
            .build(),
    );

    // Process through the handlers that the address query depends on
    process_and_commit(&tx_digests::TxDigests, &checkpoint, &ctx.db).await;
    process_and_commit(
        &tx_affected_addresses::TxAffectedAddresses,
        &checkpoint,
        &ctx.db,
    )
    .await;

    // Query sender's transactions via GraphQL
    let sender_hex = format!("0x{}", hex::encode(sender.to_inner()));
    let query = format!(
        r#"{{ address(address: "{}") {{ transactions {{ edges {{ node {{ sequenceNumber }} }} }} }} }}"#,
        sender_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let edges = json["data"]["address"]["transactions"]["edges"]
        .as_array()
        .unwrap();
    // Sender is affected in both transactions
    assert_eq!(edges.len(), 2);
}

// ---------------------------------------------------------------------------
// E2e: Object indexing → kv_objects + obj_info → GraphQL object query
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_e2e_object_to_graphql() {
    let ctx = setup().await;

    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();

    let checkpoint = Arc::new(
        TestCheckpointBuilder::new(0)
            .add_transfer_coin(sender, recipient, 5000)
            .build(),
    );

    // Process through kv_objects, obj_info, obj_versions handlers
    let kv_values =
        process_and_commit(&kv_objects::KvObjects, &checkpoint, &ctx.db).await;
    process_and_commit(&obj_info::ObjInfo, &checkpoint, &ctx.db).await;
    process_and_commit(&obj_versions::ObjVersions, &checkpoint, &ctx.db).await;

    // The kv_objects handler should have produced objects
    assert!(
        kv_values.len() >= 2,
        "Expected at least 2 objects (gas + coin), got {}",
        kv_values.len()
    );

    // Query the first object via GraphQL
    let obj_id_hex = format!("0x{}", hex::encode(&kv_values[0].object_id));
    let query = format!(
        r#"{{ object(id: "{}") {{ objectId version ownerKind }} }}"#,
        obj_id_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let obj = &json["data"]["object"];
    assert!(!obj.is_null(), "Object should be found");
    assert_eq!(obj["objectId"], obj_id_hex);
}

// ---------------------------------------------------------------------------
// E2e: Full checkpoint flow → multiple handler pipelines → verify all queryable
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_e2e_full_checkpoint_flow() {
    let ctx = setup().await;

    let sender = SomaAddress::random();
    let recipient = SomaAddress::random();

    // Build a checkpoint with a transfer and a target
    let target = test_target(0, TargetStatus::Open, 7500);
    let checkpoint = Arc::new(
        TestCheckpointBuilder::new(0)
            .add_transfer_coin(sender, recipient, 3000)
            .add_target(target)
            .build(),
    );

    // Run all relevant handlers
    process_and_commit(&kv_checkpoints::KvCheckpoints, &checkpoint, &ctx.db).await;
    process_and_commit(
        &cp_sequence_numbers::CpSequenceNumbers,
        &checkpoint,
        &ctx.db,
    )
    .await;
    process_and_commit(&kv_transactions::KvTransactions, &checkpoint, &ctx.db).await;
    process_and_commit(&tx_digests::TxDigests, &checkpoint, &ctx.db).await;
    process_and_commit(&kv_objects::KvObjects, &checkpoint, &ctx.db).await;
    process_and_commit(&obj_info::ObjInfo, &checkpoint, &ctx.db).await;
    process_and_commit(&obj_versions::ObjVersions, &checkpoint, &ctx.db).await;
    process_and_commit(
        &tx_affected_addresses::TxAffectedAddresses,
        &checkpoint,
        &ctx.db,
    )
    .await;
    process_and_commit(
        &tx_affected_objects::TxAffectedObjects,
        &checkpoint,
        &ctx.db,
    )
    .await;
    process_and_commit(&tx_kinds::TxKinds, &checkpoint, &ctx.db).await;
    let target_values =
        process_and_commit(&soma_targets::SomaTargets, &checkpoint, &ctx.db).await;

    // Verify checkpoint queryable
    let json = execute(
        &ctx.schema,
        r#"{ checkpoint(sequenceNumber: 0) { sequenceNumber epoch } }"#,
    )
    .await;
    assert_eq!(json["data"]["checkpoint"]["sequenceNumber"], "0");

    // Verify transaction queryable
    let tx = &checkpoint.transactions[0];
    let digest_b58 = bs58::encode(tx.transaction.digest().inner()).into_string();
    let query = format!(
        r#"{{ transaction(digest: "{}") {{ checkpointSequenceNumber }} }}"#,
        digest_b58
    );
    let json = execute(&ctx.schema, &query).await;
    assert!(!json["data"]["transaction"].is_null());

    // Verify target queryable
    assert!(!target_values.is_empty());
    let target_id_hex = format!("0x{}", hex::encode(&target_values[0].target_id));
    let query = format!(
        r#"{{ target(targetId: "{}") {{ status rewardPool }} }}"#,
        target_id_hex
    );
    let json = execute(&ctx.schema, &query).await;
    assert_eq!(json["data"]["target"]["status"], "open");
    assert_eq!(json["data"]["target"]["rewardPool"], "7500");

    // Verify address transactions queryable
    let sender_hex = format!("0x{}", hex::encode(sender.to_inner()));
    let query = format!(
        r#"{{ address(address: "{}") {{ transactions {{ edges {{ node {{ sequenceNumber }} }} }} }} }}"#,
        sender_hex
    );
    let json = execute(&ctx.schema, &query).await;
    let edges = json["data"]["address"]["transactions"]["edges"]
        .as_array()
        .unwrap();
    assert!(!edges.is_empty());

    // Verify chain identifier returns the encoded summary
    let json = execute(&ctx.schema, "{ chainIdentifier }").await;
    assert_ne!(json["data"]["chainIdentifier"], "unknown");
}
