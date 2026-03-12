// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! GraphQL resolver integration tests — Layer 3 of the indexer test plan.
//!
//! Each test:
//! 1. Spins up an ephemeral Postgres via TempDb
//! 2. Runs indexer-alt-schema migrations
//! 3. Seeds data via direct diesel inserts
//! 4. Builds a GraphQL schema backed by PgReader
//! 5. Executes a GraphQL query and asserts on the JSON response
//!
//! Tests are `#[ignore]` by default since they require Postgres on PATH.
//! Run with: `cargo test -p soma-graphql --test graphql_tests -- --ignored`

use std::ops::DerefMut;
use std::sync::Arc;

use diesel::prelude::*;
use diesel_async::RunQueryDsl;
use serde_json::Value;

use indexer_alt_schema::checkpoints::StoredCheckpoint;
use indexer_alt_schema::epochs::{StoredEpochEnd, StoredEpochStart};
use indexer_alt_schema::soma::{StoredModel, StoredReward, StoredTarget};
use indexer_alt_schema::transactions::{StoredTransaction, StoredTxDigest};
use indexer_pg_db::DbArgs;

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

    // Write-mode Db for seeding + migrations
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

    // Read-mode PgReader for GraphQL
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

// ---------------------------------------------------------------------------
// serviceConfig
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_service_config() {
    let ctx = setup().await;
    let json = execute(
        &ctx.schema,
        "{ serviceConfig { maxPageSize defaultPageSize maxQueryDepth } }",
    )
    .await;

    let sc = &json["data"]["serviceConfig"];
    assert_eq!(sc["maxPageSize"], 50);
    assert_eq!(sc["defaultPageSize"], 20);
    assert_eq!(sc["maxQueryDepth"], 10);
}

// ---------------------------------------------------------------------------
// chainIdentifier
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_chain_identifier_no_data() {
    let ctx = setup().await;
    let json = execute(&ctx.schema, "{ chainIdentifier }").await;
    assert_eq!(json["data"]["chainIdentifier"], "unknown");
}

#[tokio::test]
#[ignore]
async fn test_chain_identifier_with_genesis() {
    let ctx = setup().await;

    // Seed genesis checkpoint
    let summary_bytes = vec![1, 2, 3, 4];
    {
        let mut conn = ctx.db.connect().await.unwrap();
        use indexer_alt_schema::schema::kv_checkpoints;
        diesel::insert_into(kv_checkpoints::table)
            .values(&StoredCheckpoint {
                sequence_number: 0,
                checkpoint_summary: summary_bytes.clone(),
                checkpoint_contents: vec![],
                validator_signatures: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(&ctx.schema, "{ chainIdentifier }").await;
    let chain_id = json["data"]["chainIdentifier"].as_str().unwrap();
    // Should be base58 of [1,2,3,4]
    assert_eq!(chain_id, bs58::encode(&summary_bytes).into_string());
}

// ---------------------------------------------------------------------------
// checkpoint
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_checkpoint_by_sequence_number() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{cp_sequence_numbers, kv_checkpoints};

    diesel::insert_into(kv_checkpoints::table)
        .values(&StoredCheckpoint {
            sequence_number: 5,
            checkpoint_summary: vec![10, 20],
            checkpoint_contents: vec![30, 40],
            validator_signatures: vec![50, 60],
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    diesel::insert_into(cp_sequence_numbers::table)
        .values(&(
            cp_sequence_numbers::cp_sequence_number.eq(5i64),
            cp_sequence_numbers::epoch.eq(1i64),
            cp_sequence_numbers::tx_lo.eq(100i64),
        ))
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let json = execute(
        &ctx.schema,
        r#"{ checkpoint(sequenceNumber: 5) { sequenceNumber epoch txLo } }"#,
    )
    .await;

    let cp = &json["data"]["checkpoint"];
    assert_eq!(cp["sequenceNumber"], "5");
    assert_eq!(cp["epoch"], "1");
    assert_eq!(cp["txLo"], "100");
}

#[tokio::test]
#[ignore]
async fn test_checkpoint_latest() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::kv_checkpoints;

    for seq in [0i64, 1, 2] {
        diesel::insert_into(kv_checkpoints::table)
            .values(&StoredCheckpoint {
                sequence_number: seq,
                checkpoint_summary: vec![seq as u8],
                checkpoint_contents: vec![],
                validator_signatures: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ checkpoint { sequenceNumber } }"#,
    )
    .await;

    assert_eq!(json["data"]["checkpoint"]["sequenceNumber"], "2");
}

#[tokio::test]
#[ignore]
async fn test_checkpoint_not_found() {
    let ctx = setup().await;

    let json = execute(
        &ctx.schema,
        r#"{ checkpoint(sequenceNumber: 999) { sequenceNumber } }"#,
    )
    .await;

    assert!(json["data"]["checkpoint"].is_null());
}

// ---------------------------------------------------------------------------
// transaction
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_transaction_by_digest() {
    let ctx = setup().await;

    let digest = vec![0xAA; 32];
    let digest_b58 = bs58::encode(&digest).into_string();

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::kv_transactions;

    diesel::insert_into(kv_transactions::table)
        .values(&StoredTransaction {
            tx_digest: digest.clone(),
            cp_sequence_number: 7,
            timestamp_ms: 1_000_000,
            raw_transaction: vec![1, 2, 3],
            raw_effects: vec![4, 5, 6],
            events: vec![],
            user_signatures: vec![7, 8],
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let query = format!(
        r#"{{ transaction(digest: "{}") {{ checkpointSequenceNumber }} }}"#,
        digest_b58
    );
    let json = execute(&ctx.schema, &query).await;

    assert_eq!(
        json["data"]["transaction"]["checkpointSequenceNumber"],
        "7"
    );
}

#[tokio::test]
#[ignore]
async fn test_transaction_not_found() {
    let ctx = setup().await;
    let fake_digest = bs58::encode(&[0u8; 32]).into_string();
    let query = format!(
        r#"{{ transaction(digest: "{}") {{ checkpointSequenceNumber }} }}"#,
        fake_digest
    );
    let json = execute(&ctx.schema, &query).await;
    assert!(json["data"]["transaction"].is_null());
}

// ---------------------------------------------------------------------------
// epoch
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_epoch_with_start_and_end() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{kv_epoch_ends, kv_epoch_starts};

    diesel::insert_into(kv_epoch_starts::table)
        .values(&StoredEpochStart {
            epoch: 3,
            protocol_version: 1,
            cp_lo: 100,
            start_timestamp_ms: 5_000_000,
            reference_gas_price: 1000,
            system_state: vec![99],
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    diesel::insert_into(kv_epoch_ends::table)
        .values(&StoredEpochEnd {
            epoch: 3,
            cp_hi: 199,
            tx_hi: 500,
            end_timestamp_ms: 6_000_000,
            safe_mode: false,
            total_stake: Some(1_000_000),
            storage_fund_balance: None,
            storage_fund_reinvestment: None,
            storage_charge: None,
            storage_rebate: None,
            stake_subsidy_amount: None,
            total_gas_fees: Some(50_000),
            total_stake_rewards_distributed: None,
            leftover_storage_fund_inflow: None,
            epoch_commitments: vec![],
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let json = execute(
        &ctx.schema,
        r#"{ epoch(epochId: 3) { epochId protocolVersion startCheckpoint endCheckpoint safeMode totalStake totalGasFees } }"#,
    )
    .await;

    let ep = &json["data"]["epoch"];
    assert_eq!(ep["epochId"], "3");
    assert_eq!(ep["protocolVersion"], "1");
    assert_eq!(ep["startCheckpoint"], "100");
    assert_eq!(ep["endCheckpoint"], "199");
    assert_eq!(ep["safeMode"], false);
    assert_eq!(ep["totalStake"], "1000000");
    assert_eq!(ep["totalGasFees"], "50000");
}

#[tokio::test]
#[ignore]
async fn test_epoch_ongoing_no_end() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::kv_epoch_starts;

    diesel::insert_into(kv_epoch_starts::table)
        .values(&StoredEpochStart {
            epoch: 0,
            protocol_version: 1,
            cp_lo: 0,
            start_timestamp_ms: 1_000_000,
            reference_gas_price: 1000,
            system_state: vec![],
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let json = execute(
        &ctx.schema,
        r#"{ epoch(epochId: 0) { epochId endCheckpoint endTimestamp safeMode } }"#,
    )
    .await;

    let ep = &json["data"]["epoch"];
    assert_eq!(ep["epochId"], "0");
    assert!(ep["endCheckpoint"].is_null());
    assert!(ep["endTimestamp"].is_null());
    assert!(ep["safeMode"].is_null());
}

#[tokio::test]
#[ignore]
async fn test_epoch_latest() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::kv_epoch_starts;

    for e in 0..3i64 {
        diesel::insert_into(kv_epoch_starts::table)
            .values(&StoredEpochStart {
                epoch: e,
                protocol_version: 1,
                cp_lo: e * 100,
                start_timestamp_ms: 1_000_000 + e * 1000,
                reference_gas_price: 1000,
                system_state: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ epoch { epochId } }"#,
    )
    .await;

    assert_eq!(json["data"]["epoch"]["epochId"], "2");
}

// ---------------------------------------------------------------------------
// target (single lookup)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_target_lookup() {
    let ctx = setup().await;

    let target_id = vec![0xAA; 32];
    let target_hex = format!("0x{}", hex::encode(&target_id));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    diesel::insert_into(soma_targets::table)
        .values(&StoredTarget {
            target_id: target_id.clone(),
            cp_sequence_number: 10,
            epoch: 1,
            status: "Open".to_string(),
            submitter: None,
            winning_model_id: None,
            reward_pool: 5000,
            bond_amount: 0,
            report_count: 0,
            state_bcs: vec![1, 2, 3],
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let query = format!(
        r#"{{ target(targetId: "{}") {{ targetId epoch status rewardPool bondAmount reportCount }} }}"#,
        target_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let t = &json["data"]["target"];
    assert_eq!(t["targetId"], target_hex);
    assert_eq!(t["epoch"], "1");
    assert_eq!(t["status"], "Open");
    assert_eq!(t["rewardPool"], "5000");
    assert_eq!(t["bondAmount"], "0");
    assert_eq!(t["reportCount"], 0);
}

#[tokio::test]
#[ignore]
async fn test_target_returns_latest_version() {
    let ctx = setup().await;

    let target_id = vec![0xBB; 32];
    let target_hex = format!("0x{}", hex::encode(&target_id));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    // Insert two versions: Open at cp=5, Filled at cp=10
    for (cp, status) in [(5i64, "Open"), (10, "Filled")] {
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: target_id.clone(),
                cp_sequence_number: cp,
                epoch: 1,
                status: status.to_string(),
                submitter: if status == "Filled" {
                    Some(vec![0xCC; 32])
                } else {
                    None
                },
                winning_model_id: None,
                reward_pool: 5000,
                bond_amount: 0,
                report_count: 0,
                state_bcs: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(
        r#"{{ target(targetId: "{}") {{ status submitter }} }}"#,
        target_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let t = &json["data"]["target"];
    assert_eq!(t["status"], "Filled");
    assert!(!t["submitter"].is_null());
}

// ---------------------------------------------------------------------------
// targets (paginated)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_targets_pagination() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    // Insert 5 targets at different checkpoints
    for i in 0..5u8 {
        let mut id = vec![0u8; 32];
        id[0] = i;
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: id,
                cp_sequence_number: (i as i64) * 10,
                epoch: 0,
                status: "Open".to_string(),
                submitter: None,
                winning_model_id: None,
                reward_pool: 1000,
                bond_amount: 0,
                report_count: 0,
                state_bcs: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // Request first 2
    let json = execute(
        &ctx.schema,
        r#"{ targets(first: 2) { edges { cursor node { status } } pageInfo { hasNextPage hasPreviousPage } } }"#,
    )
    .await;

    let conn_data = &json["data"]["targets"];
    let edges = conn_data["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    assert_eq!(conn_data["pageInfo"]["hasNextPage"], true);
    assert_eq!(conn_data["pageInfo"]["hasPreviousPage"], false);

    // Use cursor from last edge to get next page
    let cursor = edges[1]["cursor"].as_str().unwrap();
    let query = format!(
        r#"{{ targets(first: 2, after: "{}") {{ edges {{ node {{ status }} }} pageInfo {{ hasNextPage hasPreviousPage }} }} }}"#,
        cursor
    );
    let json2 = execute(&ctx.schema, &query).await;

    let conn2 = &json2["data"]["targets"];
    let edges2 = conn2["edges"].as_array().unwrap();
    assert_eq!(edges2.len(), 2);
    assert_eq!(conn2["pageInfo"]["hasPreviousPage"], true);
}

#[tokio::test]
#[ignore]
async fn test_targets_filter_by_status() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    // Insert Open + Filled targets
    for (i, status) in [(0u8, "Open"), (1, "Filled"), (2, "Open")] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: id,
                cp_sequence_number: i as i64,
                epoch: 0,
                status: status.to_string(),
                submitter: None,
                winning_model_id: None,
                reward_pool: 1000,
                bond_amount: 0,
                report_count: 0,
                state_bcs: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ targets(filter: { status: "Open" }) { edges { node { status } } } }"#,
    )
    .await;

    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        assert_eq!(edge["node"]["status"], "Open");
    }
}

#[tokio::test]
#[ignore]
async fn test_targets_empty_result() {
    let ctx = setup().await;

    let json = execute(
        &ctx.schema,
        r#"{ targets { edges { node { status } } pageInfo { hasNextPage hasPreviousPage } } }"#,
    )
    .await;

    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert!(edges.is_empty());
    assert_eq!(json["data"]["targets"]["pageInfo"]["hasNextPage"], false);
    assert_eq!(
        json["data"]["targets"]["pageInfo"]["hasPreviousPage"],
        false
    );
}

// ---------------------------------------------------------------------------
// models (paginated)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_models_by_epoch() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    for i in 0..3u8 {
        let mut model_id = vec![0u8; 32];
        model_id[0] = i;
        diesel::insert_into(soma_models::table)
            .values(&StoredModel {
                model_id,
                epoch: 5,
                status: "active".to_string(),
                owner: vec![0xDD; 32],
                architecture_version: 1,
                commit_epoch: 2,
                stake: 100_000,
                commission_rate: 500,
                has_embedding: true,
                state_bcs: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ models(epoch: 5) { edges { node { epoch status hasEmbedding stake } } } }"#,
    )
    .await;

    let edges = json["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 3);
    for edge in edges {
        assert_eq!(edge["node"]["epoch"], "5");
        assert_eq!(edge["node"]["status"], "active");
        assert_eq!(edge["node"]["hasEmbedding"], true);
        assert_eq!(edge["node"]["stake"], "100000");
    }
}

#[tokio::test]
#[ignore]
async fn test_models_latest_epoch() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    // Models in epoch 3 and epoch 5
    for (epoch, id_byte) in [(3i64, 0xA0u8), (5, 0xB0)] {
        let mut model_id = vec![0u8; 32];
        model_id[0] = id_byte;
        diesel::insert_into(soma_models::table)
            .values(&StoredModel {
                model_id,
                epoch,
                status: "active".to_string(),
                owner: vec![0xDD; 32],
                architecture_version: 1,
                commit_epoch: 0,
                stake: 0,
                commission_rate: 0,
                has_embedding: false,
                state_bcs: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // No epoch param → returns latest (epoch 5)
    let json = execute(
        &ctx.schema,
        r#"{ models { edges { node { epoch } } } }"#,
    )
    .await;

    let edges = json["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["node"]["epoch"], "5");
}

#[tokio::test]
#[ignore]
async fn test_models_pagination() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    for i in 0..5u8 {
        let mut model_id = vec![0u8; 32];
        model_id[0] = i;
        diesel::insert_into(soma_models::table)
            .values(&StoredModel {
                model_id,
                epoch: 1,
                status: "active".to_string(),
                owner: vec![0xDD; 32],
                architecture_version: 1,
                commit_epoch: 0,
                stake: 0,
                commission_rate: 0,
                has_embedding: false,
                state_bcs: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ models(epoch: 1, first: 2) { edges { cursor node { modelId } } pageInfo { hasNextPage } } }"#,
    )
    .await;

    let edges = json["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    assert_eq!(json["data"]["models"]["pageInfo"]["hasNextPage"], true);

    // Next page
    let cursor = edges[1]["cursor"].as_str().unwrap();
    let query = format!(
        r#"{{ models(epoch: 1, first: 2, after: "{}") {{ edges {{ node {{ modelId }} }} pageInfo {{ hasNextPage }} }} }}"#,
        cursor
    );
    let json2 = execute(&ctx.schema, &query).await;
    let edges2 = json2["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges2.len(), 2);
}

// ---------------------------------------------------------------------------
// rewards
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_rewards_by_epoch() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_rewards;

    for i in 0..3u8 {
        let mut tid = vec![0u8; 32];
        tid[0] = i;
        diesel::insert_into(soma_rewards::table)
            .values(&StoredReward {
                target_id: tid,
                cp_sequence_number: i as i64,
                epoch: 2,
                tx_digest: vec![i; 32],
                balance_changes_bcs: vec![10, 20],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ rewards(epoch: 2) { targetId epoch } }"#,
    )
    .await;

    let rewards = json["data"]["rewards"].as_array().unwrap();
    assert_eq!(rewards.len(), 3);
    for r in rewards {
        assert_eq!(r["epoch"], "2");
    }
}

#[tokio::test]
#[ignore]
async fn test_rewards_filter_by_target() {
    let ctx = setup().await;

    let target_a = vec![0xAA; 32];
    let target_b = vec![0xBB; 32];
    let target_c = vec![0xCC; 32]; // third unique target
    let target_a_hex = format!("0x{}", hex::encode(&target_a));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_rewards;

    // PK is (target_id), so each target_id can only appear once
    for (tid, cp) in [(&target_a, 1i64), (&target_b, 2), (&target_c, 3)] {
        diesel::insert_into(soma_rewards::table)
            .values(&StoredReward {
                target_id: tid.clone(),
                cp_sequence_number: cp,
                epoch: 1,
                tx_digest: vec![cp as u8; 32],
                balance_changes_bcs: vec![],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(
        r#"{{ rewards(epoch: 1, targetId: "{}") {{ targetId }} }}"#,
        target_a_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let rewards = json["data"]["rewards"].as_array().unwrap();
    assert_eq!(rewards.len(), 1);
    assert_eq!(rewards[0]["targetId"], target_a_hex);
}

#[tokio::test]
#[ignore]
async fn test_rewards_empty() {
    let ctx = setup().await;
    let json = execute(&ctx.schema, r#"{ rewards(epoch: 99) { targetId } }"#).await;
    let rewards = json["data"]["rewards"].as_array().unwrap();
    assert!(rewards.is_empty());
}

// ---------------------------------------------------------------------------
// address.transactions
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_address_transactions() {
    let ctx = setup().await;

    let addr = vec![0xEE; 32];
    let addr_hex = format!("0x{}", hex::encode(&addr));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{tx_affected_addresses, tx_digests};

    // Insert 3 transactions affecting this address
    for seq in [10i64, 20, 30] {
        diesel::insert_into(tx_affected_addresses::table)
            .values(&(
                tx_affected_addresses::tx_sequence_number.eq(seq),
                tx_affected_addresses::affected.eq(&addr),
                tx_affected_addresses::sender.eq(&addr),
            ))
            .execute(conn.deref_mut())
            .await
            .unwrap();

        diesel::insert_into(tx_digests::table)
            .values(&StoredTxDigest {
                tx_sequence_number: seq,
                tx_digest: vec![seq as u8; 32],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(
        r#"{{ address(address: "{}") {{ transactions(first: 2) {{ edges {{ node {{ sequenceNumber }} }} pageInfo {{ hasNextPage }} }} }} }}"#,
        addr_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let txs = &json["data"]["address"]["transactions"];
    let edges = txs["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    assert_eq!(txs["pageInfo"]["hasNextPage"], true);
    assert_eq!(edges[0]["node"]["sequenceNumber"], "10");
    assert_eq!(edges[1]["node"]["sequenceNumber"], "20");
}

// ---------------------------------------------------------------------------
// Schema SDL export (snapshot-style test)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_schema_sdl_has_expected_types() {
    let ctx = setup().await;
    let sdl = ctx.schema.sdl();

    // Verify core types are present
    assert!(sdl.contains("type Query"));
    assert!(sdl.contains("type Checkpoint"));
    assert!(sdl.contains("type Transaction"));
    assert!(sdl.contains("type Epoch"));
    assert!(sdl.contains("type Target"));
    assert!(sdl.contains("type Model"));
    assert!(sdl.contains("type Reward"));
    assert!(sdl.contains("type ServiceConfig"));
    assert!(sdl.contains("type Address"));

    // Verify core queries
    assert!(sdl.contains("chainIdentifier"));
    assert!(sdl.contains("serviceConfig"));
    assert!(sdl.contains("checkpoint"));
    assert!(sdl.contains("transaction"));
    assert!(sdl.contains("epoch"));
    assert!(sdl.contains("target"));
    assert!(sdl.contains("targets"));
    assert!(sdl.contains("models"));
    assert!(sdl.contains("rewards"));

    // Verify custom scalars
    assert!(sdl.contains("scalar Base64"));
    assert!(sdl.contains("scalar BigInt"));
    assert!(sdl.contains("scalar SomaAddress"));
    assert!(sdl.contains("scalar Digest"));
    assert!(sdl.contains("scalar DateTime"));
}
