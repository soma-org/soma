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
use types::test_checkpoint_data_builder::TestCheckpointBuilder;

use indexer_alt::handlers::*;
use soma_graphql::config::GraphQlConfig;
use soma_graphql::db::PgReader;
use soma_graphql::subscriptions::SubscriptionChannels;
use soma_graphql::{SomaSchema, build_schema};

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

    db.run_migrations(Some(&indexer_alt_schema::MIGRATIONS)).await.expect("migrations");

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
    let schema = build_schema(pg, config, None, SubscriptionChannels::new(16));

    TestContext { schema, db, _temp: temp }
}

async fn execute(schema: &SomaSchema, query: &str) -> Value {
    let resp = schema.execute(query).await;
    let json = serde_json::to_value(&resp).unwrap();
    assert!(resp.errors.is_empty(), "GraphQL errors: {:?}", resp.errors);
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
    let checkpoint =
        Arc::new(TestCheckpointBuilder::new(0).add_transfer_coin(sender, recipient, 5000).build());

    // Process and commit through indexer handlers
    process_and_commit(&kv_transactions::KvTransactions, &checkpoint, &ctx.db).await;
    process_and_commit(&tx_digests::TxDigests, &checkpoint, &ctx.db).await;
    process_and_commit(&kv_checkpoints::KvCheckpoints, &checkpoint, &ctx.db).await;
    process_and_commit(&cp_sequence_numbers::CpSequenceNumbers, &checkpoint, &ctx.db).await;

    // Verify via GraphQL: fetch the transaction by its digest
    let tx = &checkpoint.transactions[0];
    let digest_b58 = bs58::encode(tx.transaction.digest().inner()).into_string();

    let query =
        format!(r#"{{ transaction(digest: "{}") {{ checkpointSequenceNumber }} }}"#, digest_b58);
    let json = execute(&ctx.schema, &query).await;
    assert_eq!(json["data"]["transaction"]["checkpointSequenceNumber"], "0");

    // Verify checkpoint query works too
    let json =
        execute(&ctx.schema, r#"{ checkpoint(sequenceNumber: 0) { sequenceNumber epoch } }"#).await;
    assert_eq!(json["data"]["checkpoint"]["sequenceNumber"], "0");
    assert_eq!(json["data"]["checkpoint"]["epoch"], "0");
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
    process_and_commit(&tx_affected_addresses::TxAffectedAddresses, &checkpoint, &ctx.db).await;

    // Query sender's transactions via GraphQL
    let sender_hex = format!("0x{}", hex::encode(sender.to_inner()));
    let query = format!(
        r#"{{ address(address: "{}") {{ transactions {{ edges {{ node {{ sequenceNumber }} }} }} }} }}"#,
        sender_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let edges = json["data"]["address"]["transactions"]["edges"].as_array().unwrap();
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

    let checkpoint =
        Arc::new(TestCheckpointBuilder::new(0).add_transfer_coin(sender, recipient, 5000).build());

    // Process through kv_objects, obj_info, obj_versions handlers
    let kv_values = process_and_commit(&kv_objects::KvObjects, &checkpoint, &ctx.db).await;
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
    let query = format!(r#"{{ object(id: "{}") {{ objectId version ownerKind }} }}"#, obj_id_hex);
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

    // Build a checkpoint with a transfer
    let checkpoint = Arc::new(
        TestCheckpointBuilder::new(0)
            .add_transfer_coin(sender, recipient, 3000)
            .build(),
    );

    // Run all relevant handlers
    process_and_commit(&kv_checkpoints::KvCheckpoints, &checkpoint, &ctx.db).await;
    process_and_commit(&cp_sequence_numbers::CpSequenceNumbers, &checkpoint, &ctx.db).await;
    process_and_commit(&kv_transactions::KvTransactions, &checkpoint, &ctx.db).await;
    process_and_commit(&tx_digests::TxDigests, &checkpoint, &ctx.db).await;
    process_and_commit(&kv_objects::KvObjects, &checkpoint, &ctx.db).await;
    process_and_commit(&obj_info::ObjInfo, &checkpoint, &ctx.db).await;
    process_and_commit(&obj_versions::ObjVersions, &checkpoint, &ctx.db).await;
    process_and_commit(&tx_affected_addresses::TxAffectedAddresses, &checkpoint, &ctx.db).await;
    process_and_commit(&tx_affected_objects::TxAffectedObjects, &checkpoint, &ctx.db).await;
    process_and_commit(&tx_kinds::TxKinds, &checkpoint, &ctx.db).await;

    // Verify checkpoint queryable
    let json =
        execute(&ctx.schema, r#"{ checkpoint(sequenceNumber: 0) { sequenceNumber epoch } }"#).await;
    assert_eq!(json["data"]["checkpoint"]["sequenceNumber"], "0");

    // Verify transaction queryable
    let tx = &checkpoint.transactions[0];
    let digest_b58 = bs58::encode(tx.transaction.digest().inner()).into_string();
    let query =
        format!(r#"{{ transaction(digest: "{}") {{ checkpointSequenceNumber }} }}"#, digest_b58);
    let json = execute(&ctx.schema, &query).await;
    assert!(!json["data"]["transaction"].is_null());

    // Verify address transactions queryable
    let sender_hex = format!("0x{}", hex::encode(sender.to_inner()));
    let query = format!(
        r#"{{ address(address: "{}") {{ transactions {{ edges {{ node {{ sequenceNumber }} }} }} }} }}"#,
        sender_hex
    );
    let json = execute(&ctx.schema, &query).await;
    let edges = json["data"]["address"]["transactions"]["edges"].as_array().unwrap();
    assert!(!edges.is_empty());

    // Verify chain identifier returns the encoded summary
    let json = execute(&ctx.schema, "{ chainIdentifier }").await;
    assert_ne!(json["data"]["chainIdentifier"], "unknown");
}
