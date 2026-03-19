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
use indexer_alt_schema::cp_sequence_numbers::StoredCpSequenceNumbers;
use indexer_alt_schema::epochs::{StoredEpochEnd, StoredEpochStart};
use indexer_alt_schema::soma::{
    StoredEpochState, StoredModel, StoredReward, StoredRewardBalance, StoredStakedSoma,
    StoredTarget, StoredTargetReport,
};
use indexer_alt_schema::transactions::{StoredTransaction, StoredTxDigest};
use indexer_pg_db::DbArgs;

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

    db.run_migrations(Some(&indexer_alt_schema::MIGRATIONS)).await.expect("migrations");

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
    let schema = build_schema(pg, config, None, SubscriptionChannels::new(16));

    TestContext { schema, db, _temp: temp }
}

fn test_model(model_id: Vec<u8>, epoch: i64) -> StoredModel {
    StoredModel {
        model_id,
        epoch,
        status: "active".to_string(),
        owner: vec![0xDD; 32],
        architecture_version: 1,
        commit_epoch: 0,
        stake: 0,
        commission_rate: 0,
        next_epoch_commission_rate: 0,
        staking_pool_id: vec![0; 32],
        activation_epoch: None,
        deactivation_epoch: None,
        rewards_pool: 0,
        pool_token_balance: 0,
        pending_stake: 0,
        pending_total_soma_withdraw: 0,
        pending_pool_token_withdraw: 0,
        exchange_rates_json: "{}".to_string(),
        manifest_url: None,
        manifest_checksum: None,
        manifest_size: None,
        weights_commitment: None,
        has_pending_update: false,
        pending_manifest_url: None,
        pending_manifest_checksum: None,
        pending_manifest_size: None,
        pending_weights_commitment: None,
        pending_commit_epoch: None,
    }
}

async fn execute(schema: &SomaSchema, query: &str) -> Value {
    let resp = schema.execute(query).await;
    let json = serde_json::to_value(&resp).unwrap();
    assert!(resp.errors.is_empty(), "GraphQL errors: {:?}", resp.errors);
    json
}

// ---------------------------------------------------------------------------
// serviceConfig
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_service_config() {
    let ctx = setup().await;
    let json =
        execute(&ctx.schema, "{ serviceConfig { maxPageSize defaultPageSize maxQueryDepth } }")
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

    let json =
        execute(&ctx.schema, r#"{ checkpoint(sequenceNumber: 5) { sequenceNumber epoch txLo } }"#)
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
    use indexer_alt_schema::schema::{cp_sequence_numbers, kv_checkpoints};

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
        diesel::insert_into(cp_sequence_numbers::table)
            .values(&StoredCpSequenceNumbers { cp_sequence_number: seq, tx_lo: seq * 10, epoch: 0 })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(&ctx.schema, r#"{ checkpoint { sequenceNumber } }"#).await;

    assert_eq!(json["data"]["checkpoint"]["sequenceNumber"], "2");
}

#[tokio::test]
#[ignore]
async fn test_checkpoint_not_found() {
    let ctx = setup().await;

    let json =
        execute(&ctx.schema, r#"{ checkpoint(sequenceNumber: 999) { sequenceNumber } }"#).await;

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

    let query =
        format!(r#"{{ transaction(digest: "{}") {{ checkpointSequenceNumber }} }}"#, digest_b58);
    let json = execute(&ctx.schema, &query).await;

    assert_eq!(json["data"]["transaction"]["checkpointSequenceNumber"], "7");
}

#[tokio::test]
#[ignore]
async fn test_transaction_not_found() {
    let ctx = setup().await;
    let fake_digest = bs58::encode(&[0u8; 32]).into_string();
    let query =
        format!(r#"{{ transaction(digest: "{}") {{ checkpointSequenceNumber }} }}"#, fake_digest);
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
    use indexer_alt_schema::schema::{cp_sequence_numbers, kv_epoch_starts};

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
        // Seed cp_sequence_numbers so "latest epoch" discovery works.
        diesel::insert_into(cp_sequence_numbers::table)
            .values(&StoredCpSequenceNumbers { cp_sequence_number: e * 100, tx_lo: 0, epoch: e })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(&ctx.schema, r#"{ epoch { epochId } }"#).await;

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

            winning_distance_score: None,
            winning_loss_score: None,
            winning_model_owner: None,
            fill_epoch: None,
            distance_threshold: 0.5,
            model_ids_json: "[]".to_string(),
            winning_data_url: None,
            winning_data_checksum: None,
            winning_data_size: None,
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
                submitter: if status == "Filled" { Some(vec![0xCC; 32]) } else { None },
                winning_model_id: None,
                reward_pool: 5000,
                bond_amount: 0,
                report_count: 0,

                winning_distance_score: if status == "Filled" { Some(0.123) } else { None },
                winning_loss_score: if status == "Filled" { Some(0.456) } else { None },
                winning_model_owner: None,
                fill_epoch: if status == "Filled" { Some(1) } else { None },
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(r#"{{ target(targetId: "{}") {{ status submitter }} }}"#, target_hex);
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

                winning_distance_score: None,
                winning_loss_score: None,
                winning_model_owner: None,
                fill_epoch: None,
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
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

                winning_distance_score: None,
                winning_loss_score: None,
                winning_model_owner: None,
                fill_epoch: None,
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
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
    assert_eq!(json["data"]["targets"]["pageInfo"]["hasPreviousPage"], false);
}

// ---------------------------------------------------------------------------
// targets: new denormalized fields
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_target_denormalized_scores() {
    let ctx = setup().await;

    let target_id = vec![0xDD; 32];
    let target_hex = format!("0x{}", hex::encode(&target_id));
    let model_owner = vec![0xEE; 32];
    let model_owner_hex = format!("0x{}", hex::encode(&model_owner));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    let model_a = vec![0x11; 32];
    let model_b = vec![0x22; 32];
    let model_a_hex = format!("0x{}", hex::encode(&model_a));
    let model_b_hex = format!("0x{}", hex::encode(&model_b));
    let model_ids_json = serde_json::to_string(&vec![&model_a_hex, &model_b_hex]).unwrap();

    diesel::insert_into(soma_targets::table)
        .values(&StoredTarget {
            target_id: target_id.clone(),
            cp_sequence_number: 20,
            epoch: 3,
            status: "Filled".to_string(),
            submitter: Some(vec![0xAA; 32]),
            winning_model_id: Some(vec![0xBB; 32]),
            reward_pool: 10000,
            bond_amount: 500,
            report_count: 0,

            winning_distance_score: Some(0.15),
            winning_loss_score: Some(0.042),
            winning_model_owner: Some(model_owner.clone()),
            fill_epoch: Some(3),
            distance_threshold: 0.3,
            model_ids_json,
            winning_data_url: Some("https://data.example.com/submission.tar.gz".to_string()),
            winning_data_checksum: Some(vec![0xAB; 32]),
            winning_data_size: Some(1024),
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let query = format!(
        r#"{{ target(targetId: "{}") {{ winningDistanceScore winningLossScore winningModelOwner fillEpoch distanceThreshold modelIds winningDataUrl winningDataSize }} }}"#,
        target_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let t = &json["data"]["target"];
    assert!((t["winningDistanceScore"].as_f64().unwrap() - 0.15).abs() < 1e-6);
    assert!((t["winningLossScore"].as_f64().unwrap() - 0.042).abs() < 1e-6);
    assert_eq!(t["winningModelOwner"], model_owner_hex);
    assert_eq!(t["fillEpoch"], "3");
    assert!((t["distanceThreshold"].as_f64().unwrap() - 0.3).abs() < 1e-6);
    let model_ids = t["modelIds"].as_array().unwrap();
    assert_eq!(model_ids.len(), 2);
    assert_eq!(model_ids[0], model_a_hex);
    assert_eq!(model_ids[1], model_b_hex);
    assert_eq!(t["winningDataUrl"], "https://data.example.com/submission.tar.gz");
    assert_eq!(t["winningDataSize"], "1024");
}

#[tokio::test]
#[ignore]
async fn test_target_null_scores_when_open() {
    let ctx = setup().await;

    let target_id = vec![0xFF; 32];
    let target_hex = format!("0x{}", hex::encode(&target_id));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    diesel::insert_into(soma_targets::table)
        .values(&StoredTarget {
            target_id: target_id.clone(),
            cp_sequence_number: 1,
            epoch: 0,
            status: "Open".to_string(),
            submitter: None,
            winning_model_id: None,
            reward_pool: 5000,
            bond_amount: 0,
            report_count: 0,

            winning_distance_score: None,
            winning_loss_score: None,
            winning_model_owner: None,
            fill_epoch: None,
            distance_threshold: 0.5,
            model_ids_json: "[]".to_string(),
            winning_data_url: None,
            winning_data_checksum: None,
            winning_data_size: None,
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let query = format!(
        r#"{{ target(targetId: "{}") {{ winningDistanceScore winningLossScore winningModelOwner fillEpoch }} }}"#,
        target_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let t = &json["data"]["target"];
    assert!(t["winningDistanceScore"].is_null());
    assert!(t["winningLossScore"].is_null());
    assert!(t["winningModelOwner"].is_null());
    assert!(t["fillEpoch"].is_null());
}

// ---------------------------------------------------------------------------
// targets: filter by submitter and winningModelId
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_targets_filter_by_submitter() {
    let ctx = setup().await;

    let submitter_a = vec![0xA1; 32];
    let submitter_b = vec![0xB2; 32];
    let submitter_a_hex = format!("0x{}", hex::encode(&submitter_a));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    for (i, sub) in [(0u8, &submitter_a), (1, &submitter_b), (2, &submitter_a)] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: id,
                cp_sequence_number: i as i64,
                epoch: 1,
                status: "Filled".to_string(),
                submitter: Some(sub.clone()),
                winning_model_id: Some(vec![0xCC; 32]),
                reward_pool: 1000,
                bond_amount: 100,
                report_count: 0,

                winning_distance_score: Some(0.1),
                winning_loss_score: Some(0.05),
                winning_model_owner: None,
                fill_epoch: Some(1),
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(
        r#"{{ targets(filter: {{ submitter: "{}" }}) {{ edges {{ node {{ submitter winningDistanceScore }} }} }} }}"#,
        submitter_a_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        assert_eq!(edge["node"]["submitter"], submitter_a_hex);
    }
}

#[tokio::test]
#[ignore]
async fn test_targets_filter_by_winning_model_id() {
    let ctx = setup().await;

    let model_a = vec![0xAA; 32];
    let model_b = vec![0xBB; 32];
    let model_a_hex = format!("0x{}", hex::encode(&model_a));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    for (i, model) in [(0u8, &model_a), (1, &model_b), (2, &model_a)] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: id,
                cp_sequence_number: i as i64,
                epoch: 1,
                status: "Filled".to_string(),
                submitter: Some(vec![0xDD; 32]),
                winning_model_id: Some(model.clone()),
                reward_pool: 2000,
                bond_amount: 200,
                report_count: 0,

                winning_distance_score: Some(0.2),
                winning_loss_score: Some(0.08),
                winning_model_owner: None,
                fill_epoch: Some(1),
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(
        r#"{{ targets(filter: {{ winningModelId: "{}" }}) {{ edges {{ node {{ winningModelId winningLossScore }} }} }} }}"#,
        model_a_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        assert_eq!(edge["node"]["winningModelId"], model_a_hex);
        assert!((edge["node"]["winningLossScore"].as_f64().unwrap() - 0.08).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// targets: filter by fillEpoch and winningModelOwner
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_targets_filter_by_fill_epoch() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    for (i, fe) in [(0u8, 5i64), (1, 6), (2, 5)] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: id,
                cp_sequence_number: i as i64,
                epoch: 1,
                status: "Filled".to_string(),
                submitter: Some(vec![0xDD; 32]),
                winning_model_id: Some(vec![0xCC; 32]),
                reward_pool: 1000,
                bond_amount: 100,
                report_count: 0,

                winning_distance_score: Some(0.1),
                winning_loss_score: Some(0.05),
                winning_model_owner: Some(vec![0xEE; 32]),
                fill_epoch: Some(fe),
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ targets(filter: { fillEpoch: 5 }) { edges { node { fillEpoch } } } }"#,
    )
    .await;

    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        assert_eq!(edge["node"]["fillEpoch"], "5");
    }
}

#[tokio::test]
#[ignore]
async fn test_targets_filter_by_winning_model_owner() {
    let ctx = setup().await;

    let owner_a = vec![0xA1; 32];
    let owner_b = vec![0xB2; 32];
    let owner_a_hex = format!("0x{}", hex::encode(&owner_a));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    for (i, owner) in [(0u8, &owner_a), (1, &owner_b), (2, &owner_a)] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: id,
                cp_sequence_number: i as i64,
                epoch: 1,
                status: "Filled".to_string(),
                submitter: Some(vec![0xDD; 32]),
                winning_model_id: Some(vec![0xCC; 32]),
                reward_pool: 1000,
                bond_amount: 100,
                report_count: 0,

                winning_distance_score: Some(0.1),
                winning_loss_score: Some(0.05),
                winning_model_owner: Some(owner.clone()),
                fill_epoch: Some(1),
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(
        r#"{{ targets(filter: {{ winningModelOwner: "{}" }}) {{ edges {{ node {{ winningModelOwner }} }} }} }}"#,
        owner_a_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        assert_eq!(edge["node"]["winningModelOwner"], owner_a_hex);
    }
}

#[tokio::test]
#[ignore]
async fn test_targets_combined_filters() {
    let ctx = setup().await;

    let model_a = vec![0xAA; 32];
    let model_a_hex = format!("0x{}", hex::encode(&model_a));
    let submitter = vec![0xDD; 32];

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    // 3 targets: model_a in epochs 5 and 6, model_b in epoch 5
    for (i, model, fe) in [(0u8, &model_a, 5i64), (1, &vec![0xBB; 32], 5), (2, &model_a, 6)] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: id,
                cp_sequence_number: i as i64,
                epoch: 1,
                status: "Filled".to_string(),
                submitter: Some(submitter.clone()),
                winning_model_id: Some(model.clone()),
                reward_pool: 1000,
                bond_amount: 100,
                report_count: 0,

                winning_distance_score: Some(0.1),
                winning_loss_score: Some(0.05),
                winning_model_owner: None,
                fill_epoch: Some(fe),
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // Filter: winningModelId=model_a AND fillEpoch=5 → should get 1 result
    let query = format!(
        r#"{{ targets(filter: {{ winningModelId: "{}", fillEpoch: 5 }}) {{ edges {{ node {{ winningModelId fillEpoch winningDistanceScore }} }} }} }}"#,
        model_a_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["node"]["winningModelId"], model_a_hex);
    assert_eq!(edges[0]["node"]["fillEpoch"], "5");
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
        let mut m = test_model(model_id, 5);
        m.commit_epoch = 2;
        m.stake = 100_000;
        m.commission_rate = 500;
        diesel::insert_into(soma_models::table).values(&m).execute(conn.deref_mut()).await.unwrap();
    }

    let json =
        execute(&ctx.schema, r#"{ models(epoch: 5) { edges { node { epoch status stake } } } }"#)
            .await;

    let edges = json["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 3);
    for edge in edges {
        assert_eq!(edge["node"]["epoch"], "5");
        assert_eq!(edge["node"]["status"], "active");
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
            .values(&test_model(model_id, epoch))
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // No epoch param → returns latest (epoch 5)
    let json = execute(&ctx.schema, r#"{ models { edges { node { epoch } } } }"#).await;

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
            .values(&test_model(model_id, 1))
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
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(&ctx.schema, r#"{ rewards(epoch: 2) { targetId epoch } }"#).await;

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
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(r#"{{ rewards(epoch: 1, targetId: "{}") {{ targetId }} }}"#, target_a_hex);
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
            .values(&StoredTxDigest { tx_sequence_number: seq, tx_digest: vec![seq as u8; 32] })
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

    // Verify denormalized target fields
    assert!(sdl.contains("winningDistanceScore"));
    assert!(sdl.contains("winningLossScore"));
    assert!(sdl.contains("winningModelOwner"));
    assert!(sdl.contains("fillEpoch"));
    assert!(sdl.contains("distanceThreshold"));
    assert!(sdl.contains("modelIds"));
    assert!(sdl.contains("winningDataUrl"));
    assert!(sdl.contains("winningDataChecksum"));
    assert!(sdl.contains("winningDataSize"));

    // Verify Target no longer exposes raw BCS blobs
    assert!(!sdl.contains("winningDataManifestBcs"));

    // Verify denormalized model fields
    assert!(sdl.contains("nextEpochCommissionRate"));
    assert!(sdl.contains("stakingPoolId"));
    assert!(sdl.contains("activationEpoch"));
    assert!(sdl.contains("rewardsPool"));
    assert!(sdl.contains("poolTokenBalance"));
    assert!(sdl.contains("pendingStake"));
    assert!(sdl.contains("exchangeRatesJson"));
    assert!(sdl.contains("manifestUrl"));
    assert!(sdl.contains("weightsCommitment"));
    assert!(sdl.contains("hasPendingUpdate"));

    // Verify Model no longer exposes stateBcs (Epoch still has systemStateBcs)
    // Check the Model type definition specifically
    let model_section = sdl.split("type Model ").nth(1).unwrap_or("");
    let model_block = model_section.split('}').next().unwrap_or("");
    assert!(!model_block.contains("stateBcs"));

    // Verify denormalized reward fields
    assert!(sdl.contains("type RewardBalance"));
    assert!(!sdl.contains("balanceChangesBcs"));

    // Verify model filter and single lookup
    assert!(sdl.contains("input ModelFilter"));
    assert!(sdl.contains("model("));

    // Verify custom scalars
    assert!(sdl.contains("scalar Base64"));
    assert!(sdl.contains("scalar BigInt"));
    assert!(sdl.contains("scalar SomaAddress"));
    assert!(sdl.contains("scalar Digest"));
    assert!(sdl.contains("scalar DateTime"));

    // Phase 2: New types and resolvers
    assert!(sdl.contains("type StakedSoma"));
    assert!(sdl.contains("type EpochState"));
    assert!(sdl.contains("type ModelAggregates"));
    assert!(sdl.contains("type TargetAggregates"));
    assert!(sdl.contains("type RewardAggregates"));

    // StakedSoma queries
    assert!(sdl.contains("stakedSoma("));
    assert!(sdl.contains("stakedSomas("));
    // Aggregate queries
    assert!(sdl.contains("modelAggregates"));
    assert!(sdl.contains("targetAggregates"));
    assert!(sdl.contains("rewardAggregates"));
    // Epoch state query
    assert!(sdl.contains("epochState"));
    // Balance query
    assert!(sdl.contains("balance("));
    // Model history query
    assert!(sdl.contains("modelHistory("));

    // Pending model update fields
    assert!(sdl.contains("pendingManifestUrl"));
    assert!(sdl.contains("pendingManifestChecksum"));
    assert!(sdl.contains("pendingManifestSize"));
    assert!(sdl.contains("pendingWeightsCommitment"));
    assert!(sdl.contains("pendingCommitEpoch"));

    // Target nested resolvers
    assert!(sdl.contains("reporters"));

    // Model nested resolver
    let model_section2 = sdl.split("type Model ").nth(1).unwrap_or("");
    let model_block2 = model_section2.split('}').next().unwrap_or("");
    assert!(model_block2.contains("targets"));
}

// ---------------------------------------------------------------------------
// model: single lookup
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_model_single_lookup() {
    let ctx = setup().await;

    let model_id = vec![0xAA; 32];
    let model_id_hex = format!("0x{}", hex::encode(&model_id));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    // Insert the model at epoch 3 and epoch 5
    for epoch in [3i64, 5] {
        let mut m = test_model(model_id.clone(), epoch);
        m.stake = epoch * 1000;
        diesel::insert_into(soma_models::table).values(&m).execute(conn.deref_mut()).await.unwrap();
    }

    // Lookup with specific epoch
    let query =
        format!(r#"{{ model(modelId: "{}", epoch: 3) {{ modelId epoch stake }} }}"#, model_id_hex);
    let json = execute(&ctx.schema, &query).await;
    let model = &json["data"]["model"];
    assert_eq!(model["modelId"], model_id_hex);
    assert_eq!(model["epoch"], "3");
    assert_eq!(model["stake"], "3000");

    // Lookup without epoch → latest (epoch 5)
    let query = format!(r#"{{ model(modelId: "{}") {{ epoch stake }} }}"#, model_id_hex);
    let json = execute(&ctx.schema, &query).await;
    let model = &json["data"]["model"];
    assert_eq!(model["epoch"], "5");
    assert_eq!(model["stake"], "5000");

    // Lookup non-existent model
    let fake_hex = format!("0x{}", hex::encode(vec![0xFF; 32]));
    let query = format!(r#"{{ model(modelId: "{}") {{ epoch }} }}"#, fake_hex);
    let json = execute(&ctx.schema, &query).await;
    assert!(json["data"]["model"].is_null());
}

// ---------------------------------------------------------------------------
// model: denormalized fields
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_model_denormalized_fields() {
    let ctx = setup().await;

    let model_id = vec![0xAB; 32];
    let model_id_hex = format!("0x{}", hex::encode(&model_id));
    let owner = vec![0xCD; 32];
    let pool_id = vec![0xEF; 32];
    let weights = vec![0x11; 32];
    let checksum = vec![0x55; 16];

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    let m = StoredModel {
        model_id: model_id.clone(),
        epoch: 10,
        status: "active".to_string(),
        owner: owner.clone(),
        architecture_version: 2,
        commit_epoch: 7,
        stake: 500_000,
        commission_rate: 1000,
        next_epoch_commission_rate: 1200,
        staking_pool_id: pool_id.clone(),
        activation_epoch: Some(3),
        deactivation_epoch: None,
        rewards_pool: 25000,
        pool_token_balance: 100000,
        pending_stake: 5000,
        pending_total_soma_withdraw: 1000,
        pending_pool_token_withdraw: 500,
        exchange_rates_json: r#"{"1":{"soma_amount":100,"pool_token_amount":100}}"#.to_string(),
        manifest_url: Some("https://example.com/model.bin".to_string()),
        manifest_checksum: Some(checksum.clone()),
        manifest_size: Some(1024000),
        weights_commitment: Some(weights.clone()),
        has_pending_update: true,
        pending_manifest_url: None,
        pending_manifest_checksum: None,
        pending_manifest_size: None,
        pending_weights_commitment: None,
        pending_commit_epoch: None,
    };
    diesel::insert_into(soma_models::table).values(&m).execute(conn.deref_mut()).await.unwrap();

    let query = format!(
        r#"{{ model(modelId: "{}") {{
            modelId epoch status owner architectureVersion commitEpoch
            stake commissionRate nextEpochCommissionRate
            stakingPoolId activationEpoch deactivationEpoch rewardsPool
            poolTokenBalance pendingStake pendingTotalSomaWithdraw
            pendingPoolTokenWithdraw exchangeRatesJson
            manifestUrl manifestChecksum manifestSize
            weightsCommitment hasPendingUpdate
        }} }}"#,
        model_id_hex
    );
    let json = execute(&ctx.schema, &query).await;
    let m = &json["data"]["model"];

    assert_eq!(m["modelId"], model_id_hex);
    assert_eq!(m["epoch"], "10");
    assert_eq!(m["status"], "active");
    assert_eq!(m["owner"], format!("0x{}", hex::encode(&owner)));
    assert_eq!(m["architectureVersion"], "2");
    assert_eq!(m["commitEpoch"], "7");
    assert_eq!(m["stake"], "500000");
    assert_eq!(m["commissionRate"], "1000");
    assert_eq!(m["nextEpochCommissionRate"], "1200");
    assert_eq!(m["stakingPoolId"], format!("0x{}", hex::encode(&pool_id)));
    assert_eq!(m["activationEpoch"], "3");
    assert!(m["deactivationEpoch"].is_null());
    assert_eq!(m["rewardsPool"], "25000");
    assert_eq!(m["poolTokenBalance"], "100000");
    assert_eq!(m["pendingStake"], "5000");
    assert_eq!(m["pendingTotalSomaWithdraw"], "1000");
    assert_eq!(m["pendingPoolTokenWithdraw"], "500");
    assert!(m["exchangeRatesJson"].as_str().unwrap().contains("soma_amount"));
    assert_eq!(m["manifestUrl"], "https://example.com/model.bin");

    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;
    assert_eq!(m["manifestChecksum"], b64.encode(&checksum));
    assert_eq!(m["manifestSize"], "1024000");
    assert_eq!(m["weightsCommitment"], b64.encode(&weights));
    assert_eq!(m["hasPendingUpdate"], true);
}

// ---------------------------------------------------------------------------
// models: filter by status
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_model_filter_by_status() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    for (i, status) in [(0u8, "active"), (1, "pending"), (2, "active"), (3, "inactive")] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        let mut m = test_model(id, 1);
        m.status = status.to_string();
        diesel::insert_into(soma_models::table).values(&m).execute(conn.deref_mut()).await.unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ models(epoch: 1, filter: { status: "active" }) { edges { node { status } } } }"#,
    )
    .await;

    let edges = json["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        assert_eq!(edge["node"]["status"], "active");
    }
}

// ---------------------------------------------------------------------------
// models: filter by owner
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_model_filter_by_owner() {
    let ctx = setup().await;

    let owner_a = vec![0xA1; 32];
    let owner_b = vec![0xB2; 32];
    let owner_a_hex = format!("0x{}", hex::encode(&owner_a));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    for (i, owner) in [(0u8, &owner_a), (1, &owner_b), (2, &owner_a)] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        let mut m = test_model(id, 1);
        m.owner = owner.clone();
        diesel::insert_into(soma_models::table).values(&m).execute(conn.deref_mut()).await.unwrap();
    }

    let query = format!(
        r#"{{ models(epoch: 1, filter: {{ owner: "{}" }}) {{ edges {{ node {{ owner }} }} }} }}"#,
        owner_a_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let edges = json["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        assert_eq!(edge["node"]["owner"], owner_a_hex);
    }
}

// ---------------------------------------------------------------------------
// models: filter by stake range
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_model_filter_by_stake_range() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    for (i, stake) in [(0u8, 100), (1, 500), (2, 1000), (3, 5000)] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        let mut m = test_model(id, 1);
        m.stake = stake;
        diesel::insert_into(soma_models::table).values(&m).execute(conn.deref_mut()).await.unwrap();
    }

    // min_stake only
    let json = execute(
        &ctx.schema,
        r#"{ models(epoch: 1, filter: { minStake: 500 }) { edges { node { stake } } } }"#,
    )
    .await;
    let edges = json["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 3);

    // max_stake only
    let json = execute(
        &ctx.schema,
        r#"{ models(epoch: 1, filter: { maxStake: 1000 }) { edges { node { stake } } } }"#,
    )
    .await;
    let edges = json["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 3);

    // Both min and max
    let json = execute(
        &ctx.schema,
        r#"{ models(epoch: 1, filter: { minStake: 500, maxStake: 1000 }) { edges { node { stake } } } }"#,
    )
    .await;
    let edges = json["data"]["models"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
    for edge in edges {
        let s: i64 = edge["node"]["stake"].as_str().unwrap().parse().unwrap();
        assert!(s >= 500 && s <= 1000);
    }
}

// ---------------------------------------------------------------------------
// rewards: balance sub-rows
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_reward_balances() {
    let ctx = setup().await;

    let target = vec![0xAA; 32];
    let recipient_a = vec![0x11; 32];
    let recipient_b = vec![0x22; 32];
    let recipient_a_hex = format!("0x{}", hex::encode(&recipient_a));
    let recipient_b_hex = format!("0x{}", hex::encode(&recipient_b));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{soma_reward_balances, soma_rewards};

    // Seed reward
    diesel::insert_into(soma_rewards::table)
        .values(&StoredReward {
            target_id: target.clone(),
            cp_sequence_number: 10,
            epoch: 5,
            tx_digest: vec![0xDD; 32],
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Seed two balance rows
    for (recipient, amount) in [(&recipient_a, 1000i64), (&recipient_b, -200)] {
        diesel::insert_into(soma_reward_balances::table)
            .values(&StoredRewardBalance {
                target_id: target.clone(),
                cp_sequence_number: 10,
                epoch: 5,
                tx_digest: vec![0xDD; 32],
                recipient: recipient.clone(),
                amount,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json =
        execute(&ctx.schema, r#"{ rewards(epoch: 5) { targetId balances { recipient amount } } }"#)
            .await;

    let rewards = json["data"]["rewards"].as_array().unwrap();
    assert_eq!(rewards.len(), 1);
    let balances = rewards[0]["balances"].as_array().unwrap();
    assert_eq!(balances.len(), 2);

    // Check both balances are present (order may vary)
    let recipients: Vec<&str> = balances.iter().map(|b| b["recipient"].as_str().unwrap()).collect();
    assert!(recipients.contains(&recipient_a_hex.as_str()));
    assert!(recipients.contains(&recipient_b_hex.as_str()));

    // Check the positive amount
    let a_bal =
        balances.iter().find(|b| b["recipient"].as_str().unwrap() == recipient_a_hex).unwrap();
    assert_eq!(a_bal["amount"], "1000");

    // Check the negative amount
    let b_bal =
        balances.iter().find(|b| b["recipient"].as_str().unwrap() == recipient_b_hex).unwrap();
    assert_eq!(b_bal["amount"], "-200");
}

// ---------------------------------------------------------------------------
// rewards: filter by recipient
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_reward_filter_by_recipient() {
    let ctx = setup().await;

    let target_a = vec![0xAA; 32];
    let target_b = vec![0xBB; 32];
    let recipient = vec![0x11; 32];
    let recipient_hex = format!("0x{}", hex::encode(&recipient));
    let other_recipient = vec![0x22; 32];
    let target_a_hex = format!("0x{}", hex::encode(&target_a));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{soma_reward_balances, soma_rewards};

    // Two rewards in epoch 3
    for (tid, cp) in [(&target_a, 10i64), (&target_b, 20)] {
        diesel::insert_into(soma_rewards::table)
            .values(&StoredReward {
                target_id: tid.clone(),
                cp_sequence_number: cp,
                epoch: 3,
                tx_digest: vec![cp as u8; 32],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // recipient receives from target_a only; target_b goes to other_recipient
    diesel::insert_into(soma_reward_balances::table)
        .values(&StoredRewardBalance {
            target_id: target_a.clone(),
            cp_sequence_number: 10,
            epoch: 3,
            tx_digest: vec![10; 32],
            recipient: recipient.clone(),
            amount: 500,
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    diesel::insert_into(soma_reward_balances::table)
        .values(&StoredRewardBalance {
            target_id: target_b.clone(),
            cp_sequence_number: 20,
            epoch: 3,
            tx_digest: vec![20; 32],
            recipient: other_recipient.clone(),
            amount: 300,
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let query =
        format!(r#"{{ rewards(epoch: 3, recipient: "{}") {{ targetId }} }}"#, recipient_hex);
    let json = execute(&ctx.schema, &query).await;

    let rewards = json["data"]["rewards"].as_array().unwrap();
    assert_eq!(rewards.len(), 1);
    assert_eq!(rewards[0]["targetId"], target_a_hex);
}

// ===========================================================================
// Phase 2: New feature tests
// ===========================================================================

// ---------------------------------------------------------------------------
// StakedSoma query
// ---------------------------------------------------------------------------

fn test_staked_soma(
    id: Vec<u8>,
    owner: Vec<u8>,
    pool_id: Vec<u8>,
    principal: i64,
) -> StoredStakedSoma {
    StoredStakedSoma {
        staked_soma_id: id,
        cp_sequence_number: 1,
        owner: Some(owner),
        pool_id: Some(pool_id),
        stake_activation_epoch: Some(0),
        principal: Some(principal),
    }
}

#[tokio::test]
#[ignore]
async fn test_staked_soma_query() {
    let ctx = setup().await;

    let id_a = vec![0xAA; 32];
    let id_b = vec![0xBB; 32];
    let owner = vec![0x11; 32];
    let pool = vec![0xCC; 32];
    let id_a_hex = format!("0x{}", hex::encode(&id_a));
    let owner_hex = format!("0x{}", hex::encode(&owner));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_staked_soma;

    diesel::insert_into(soma_staked_soma::table)
        .values(&test_staked_soma(id_a.clone(), owner.clone(), pool.clone(), 5000))
        .execute(conn.deref_mut())
        .await
        .unwrap();

    diesel::insert_into(soma_staked_soma::table)
        .values(&test_staked_soma(id_b.clone(), owner.clone(), pool.clone(), 3000))
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Single lookup
    let query = format!(
        r#"{{ stakedSoma(id: "{}") {{ stakedSomaId owner poolId stakeActivationEpoch principal }} }}"#,
        id_a_hex
    );
    let json = execute(&ctx.schema, &query).await;
    let s = &json["data"]["stakedSoma"];
    assert_eq!(s["stakedSomaId"], id_a_hex);
    assert_eq!(s["principal"], "5000");

    // List by owner
    let query = format!(
        r#"{{ stakedSomas(owner: "{}") {{ edges {{ node {{ stakedSomaId principal }} }} }} }}"#,
        owner_hex
    );
    let json = execute(&ctx.schema, &query).await;
    let edges = json["data"]["stakedSomas"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
}

#[tokio::test]
#[ignore]
async fn test_staked_soma_tombstone() {
    let ctx = setup().await;

    let id = vec![0xAA; 32];
    let id_hex = format!("0x{}", hex::encode(&id));
    let owner = vec![0x11; 32];
    let owner_hex = format!("0x{}", hex::encode(&owner));
    let pool = vec![0xCC; 32];

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_staked_soma;

    // Live row at cp=1
    diesel::insert_into(soma_staked_soma::table)
        .values(&test_staked_soma(id.clone(), owner.clone(), pool.clone(), 5000))
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Tombstone at cp=2 (all fields NULL except id/cp)
    diesel::insert_into(soma_staked_soma::table)
        .values(&StoredStakedSoma {
            staked_soma_id: id.clone(),
            cp_sequence_number: 2,
            owner: None,
            pool_id: None,
            stake_activation_epoch: None,
            principal: None,
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Single lookup should return null (tombstone supersedes)
    let query = format!(r#"{{ stakedSoma(id: "{}") {{ stakedSomaId }} }}"#, id_hex);
    let json = execute(&ctx.schema, &query).await;
    assert!(json["data"]["stakedSoma"].is_null());

    // List by owner should return 0 edges
    let query = format!(
        r#"{{ stakedSomas(owner: "{}") {{ edges {{ node {{ stakedSomaId }} }} }} }}"#,
        owner_hex
    );
    let json = execute(&ctx.schema, &query).await;
    let edges = json["data"]["stakedSomas"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 0);
}

// ---------------------------------------------------------------------------
// Balance query
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_balance_query() {
    let ctx = setup().await;

    let addr = vec![0x11; 32];
    let addr_hex = format!("0x{}", hex::encode(&addr));

    let obj_a = vec![0xA1; 32];
    let obj_b = vec![0xB2; 32];

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{kv_objects, obj_info, obj_versions};

    // Seed obj_info: two Coin objects owned by addr
    for (oid, cp) in [(&obj_a, 1i64), (&obj_b, 2)] {
        diesel::insert_into(obj_info::table)
            .values(&(
                obj_info::object_id.eq(oid),
                obj_info::cp_sequence_number.eq(cp),
                obj_info::owner_kind.eq(Some(1i16)),
                obj_info::owner_id.eq(Some(&addr)),
                obj_info::module.eq(Some("Coin")),
                obj_info::name.eq(Some("Coin")),
            ))
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // Seed obj_versions
    for (oid, ver, cp) in [(&obj_a, 1i64, 1i64), (&obj_b, 1, 2)] {
        diesel::insert_into(obj_versions::table)
            .values(&(
                obj_versions::object_id.eq(oid),
                obj_versions::object_version.eq(ver),
                obj_versions::cp_sequence_number.eq(cp),
            ))
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // Seed kv_objects with BCS-serialized Object (matches indexer kv_objects handler)
    for (oid, ver, balance) in [(&obj_a, 1i64, 5000u64), (&obj_b, 1, 3000)] {
        use types::digests::TransactionDigest;
        use types::object::{Object, ObjectID, Owner};

        let object_id = ObjectID::from_bytes(oid.as_slice()).unwrap();
        let owner = Owner::AddressOwner(types::base::SomaAddress::from_bytes(&addr).unwrap());
        let obj = Object::new_coin(object_id, balance, owner, TransactionDigest::ZERO);
        let serialized = bcs::to_bytes(&obj).unwrap();
        diesel::insert_into(kv_objects::table)
            .values(&(
                kv_objects::object_id.eq(oid),
                kv_objects::object_version.eq(ver),
                kv_objects::serialized_object.eq(Some(&serialized)),
            ))
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(r#"{{ balance(address: "{}") }}"#, addr_hex);
    let json = execute(&ctx.schema, &query).await;
    assert_eq!(json["data"]["balance"], "8000");
}

/// Regression: coins transferred to another address must not be double-counted.
#[tokio::test]
#[ignore]
async fn test_balance_excludes_transferred_coins() {
    let ctx = setup().await;

    let addr_a = vec![0xAA; 32];
    let addr_b = vec![0xBB; 32];
    let addr_a_hex = format!("0x{}", hex::encode(&addr_a));
    let addr_b_hex = format!("0x{}", hex::encode(&addr_b));

    let coin_id = vec![0xC1; 32];

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{kv_objects, obj_info, obj_versions};

    // Coin owned by A at checkpoint 10
    diesel::insert_into(obj_info::table)
        .values(&(
            obj_info::object_id.eq(&coin_id),
            obj_info::cp_sequence_number.eq(10i64),
            obj_info::owner_kind.eq(Some(1i16)),
            obj_info::owner_id.eq(Some(&addr_a)),
            obj_info::module.eq(Some("Coin")),
            obj_info::name.eq(Some("Coin")),
        ))
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Coin transferred to B at checkpoint 20
    diesel::insert_into(obj_info::table)
        .values(&(
            obj_info::object_id.eq(&coin_id),
            obj_info::cp_sequence_number.eq(20i64),
            obj_info::owner_kind.eq(Some(1i16)),
            obj_info::owner_id.eq(Some(&addr_b)),
            obj_info::module.eq(Some("Coin")),
            obj_info::name.eq(Some("Coin")),
        ))
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // obj_versions: version 2 at checkpoint 20 (latest)
    diesel::insert_into(obj_versions::table)
        .values(&(
            obj_versions::object_id.eq(&coin_id),
            obj_versions::object_version.eq(2i64),
            obj_versions::cp_sequence_number.eq(20i64),
        ))
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // kv_objects: serialized coin owned by B with balance 5000
    {
        use types::digests::TransactionDigest;
        use types::object::{Object, ObjectID, Owner};

        let object_id = ObjectID::from_bytes(coin_id.as_slice()).unwrap();
        let owner = Owner::AddressOwner(types::base::SomaAddress::from_bytes(&addr_b).unwrap());
        let obj = Object::new_coin(object_id, 5000, owner, TransactionDigest::ZERO);
        let serialized = bcs::to_bytes(&obj).unwrap();
        diesel::insert_into(kv_objects::table)
            .values(&(
                kv_objects::object_id.eq(&coin_id),
                kv_objects::object_version.eq(2i64),
                kv_objects::serialized_object.eq(Some(&serialized)),
            ))
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // A's balance must be 0 — coin was transferred away
    let query_a = format!(r#"{{ balance(address: "{}") }}"#, addr_a_hex);
    let json_a = execute(&ctx.schema, &query_a).await;
    assert_eq!(json_a["data"]["balance"], "0");

    // B's balance must be 5000
    let query_b = format!(r#"{{ balance(address: "{}") }}"#, addr_b_hex);
    let json_b = execute(&ctx.schema, &query_b).await;
    assert_eq!(json_b["data"]["balance"], "5000");
}

// ---------------------------------------------------------------------------
// Model aggregates
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_model_aggregates() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    // 3 models at epoch 5: A (active, 10000), B (active, 20000), C (pending, 5000)
    for (i, status, stake) in
        [(0u8, "active", 10000i64), (1, "active", 20000), (2, "pending", 5000)]
    {
        let mut id = vec![0u8; 32];
        id[0] = i;
        let mut m = test_model(id, 5);
        m.status = status.to_string();
        m.stake = stake;
        diesel::insert_into(soma_models::table).values(&m).execute(conn.deref_mut()).await.unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ modelAggregates(epoch: 5) { totalCount totalStake avgStake activeCount } }"#,
    )
    .await;

    let agg = &json["data"]["modelAggregates"];
    assert_eq!(agg["totalCount"], 3);
    assert_eq!(agg["totalStake"], "35000");
    assert_eq!(agg["activeCount"], 2);
    // avg_stake = 35000 / 3 ≈ 11666.67
    let avg = agg["avgStake"].as_f64().unwrap();
    assert!((avg - 11666.67).abs() < 1.0);
}

// ---------------------------------------------------------------------------
// Target aggregates
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_target_aggregates() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    // 3 targets at epoch 1: open/filled/claimed
    for (i, status, rp) in [(0u8, "open", 1000i64), (1, "filled", 2000), (2, "claimed", 3000)] {
        let mut id = vec![0u8; 32];
        id[0] = i;
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: id,
                cp_sequence_number: i as i64,
                epoch: 1,
                status: status.to_string(),
                submitter: None,
                winning_model_id: None,
                reward_pool: rp,
                bond_amount: 0,
                report_count: 0,

                winning_distance_score: None,
                winning_loss_score: None,
                winning_model_owner: None,
                fill_epoch: None,
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json = execute(
        &ctx.schema,
        r#"{ targetAggregates(epoch: 1) { totalCount openCount filledCount claimedCount totalRewardPool } }"#,
    )
    .await;

    let agg = &json["data"]["targetAggregates"];
    assert_eq!(agg["totalCount"], 3);
    assert_eq!(agg["openCount"], 1);
    assert_eq!(agg["filledCount"], 1);
    assert_eq!(agg["claimedCount"], 1);
    assert_eq!(agg["totalRewardPool"], "6000");
}

// ---------------------------------------------------------------------------
// Epoch state
// ---------------------------------------------------------------------------

fn test_epoch_state(epoch: i64) -> StoredEpochState {
    StoredEpochState {
        epoch,
        emission_balance: 0,
        emission_per_epoch: 0,
        distance_threshold: 0.0,
        targets_generated_this_epoch: 0,
        hits_this_epoch: 0,
        hits_ema: 0,
        reward_per_target: 0,
        safe_mode: false,
        safe_mode_accumulated_fees: 0,
        safe_mode_accumulated_emissions: 0,
    }
}

#[tokio::test]
#[ignore]
async fn test_epoch_state_query() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_epoch_state;

    let mut es5 = test_epoch_state(5);
    es5.emission_balance = 1_000_000;
    es5.emission_per_epoch = 10_000;
    es5.distance_threshold = 0.85;
    es5.targets_generated_this_epoch = 50;
    es5.hits_this_epoch = 30;
    es5.hits_ema = 25;
    es5.reward_per_target = 200;
    es5.safe_mode = false;

    let es7 = StoredEpochState { epoch: 7, ..test_epoch_state(7) };

    diesel::insert_into(soma_epoch_state::table)
        .values(&es5)
        .execute(conn.deref_mut())
        .await
        .unwrap();
    diesel::insert_into(soma_epoch_state::table)
        .values(&es7)
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Query specific epoch
    let json = execute(
        &ctx.schema,
        r#"{ epochState(epoch: 5) { epoch emissionBalance emissionPerEpoch distanceThreshold targetsGeneratedThisEpoch hitsThisEpoch hitsEma rewardPerTarget safeMode safeModeAccumulatedFees safeModeAccumulatedEmissions } }"#,
    )
    .await;

    let es = &json["data"]["epochState"];
    assert_eq!(es["epoch"], "5");
    assert_eq!(es["emissionBalance"], "1000000");
    assert_eq!(es["emissionPerEpoch"], "10000");
    assert!((es["distanceThreshold"].as_f64().unwrap() - 0.85).abs() < 1e-6);
    assert_eq!(es["targetsGeneratedThisEpoch"], "50");
    assert_eq!(es["hitsThisEpoch"], "30");
    assert_eq!(es["hitsEma"], "25");
    assert_eq!(es["rewardPerTarget"], "200");
    assert_eq!(es["safeMode"], false);
    assert_eq!(es["safeModeAccumulatedFees"], "0");
    assert_eq!(es["safeModeAccumulatedEmissions"], "0");

    // Query latest (no arg) → epoch 7
    let json = execute(&ctx.schema, r#"{ epochState { epoch } }"#).await;
    assert_eq!(json["data"]["epochState"]["epoch"], "7");
}

// ---------------------------------------------------------------------------
// Pending model update fields
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_pending_model_update_fields() {
    let ctx = setup().await;

    let model_a = vec![0xAA; 32];
    let model_b = vec![0xBB; 32];
    let model_a_hex = format!("0x{}", hex::encode(&model_a));
    let model_b_hex = format!("0x{}", hex::encode(&model_b));
    let pending_checksum = vec![0xFF; 32];
    let pending_wc = vec![0x11; 32];

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    // Model A: has pending update
    let mut ma = test_model(model_a.clone(), 5);
    ma.has_pending_update = true;
    ma.pending_manifest_url = Some("https://example.com/pending.bin".to_string());
    ma.pending_manifest_checksum = Some(pending_checksum.clone());
    ma.pending_manifest_size = Some(99999);
    ma.pending_weights_commitment = Some(pending_wc.clone());
    ma.pending_commit_epoch = Some(6);

    diesel::insert_into(soma_models::table).values(&ma).execute(conn.deref_mut()).await.unwrap();

    // Model B: no pending update
    diesel::insert_into(soma_models::table)
        .values(&test_model(model_b.clone(), 5))
        .execute(conn.deref_mut())
        .await
        .unwrap();

    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD;

    // Query model A
    let query = format!(
        r#"{{ model(modelId: "{}") {{ hasPendingUpdate pendingManifestUrl pendingManifestChecksum pendingManifestSize pendingWeightsCommitment pendingCommitEpoch }} }}"#,
        model_a_hex
    );
    let json = execute(&ctx.schema, &query).await;
    let m = &json["data"]["model"];
    assert_eq!(m["hasPendingUpdate"], true);
    assert_eq!(m["pendingManifestUrl"], "https://example.com/pending.bin");
    assert_eq!(m["pendingManifestChecksum"], b64.encode(&pending_checksum));
    assert_eq!(m["pendingManifestSize"], "99999");
    assert_eq!(m["pendingWeightsCommitment"], b64.encode(&pending_wc));
    assert_eq!(m["pendingCommitEpoch"], "6");

    // Query model B: all pending fields null
    let query = format!(
        r#"{{ model(modelId: "{}") {{ hasPendingUpdate pendingManifestUrl pendingManifestChecksum pendingManifestSize pendingWeightsCommitment pendingCommitEpoch }} }}"#,
        model_b_hex
    );
    let json = execute(&ctx.schema, &query).await;
    let m = &json["data"]["model"];
    assert_eq!(m["hasPendingUpdate"], false);
    assert!(m["pendingManifestUrl"].is_null());
    assert!(m["pendingManifestChecksum"].is_null());
    assert!(m["pendingManifestSize"].is_null());
    assert!(m["pendingWeightsCommitment"].is_null());
    assert!(m["pendingCommitEpoch"].is_null());
}

// ---------------------------------------------------------------------------
// Target reporters
// ---------------------------------------------------------------------------

fn test_target_report(target_id: Vec<u8>, reporter: Vec<u8>) -> StoredTargetReport {
    StoredTargetReport { target_id, cp_sequence_number: 5, reporter }
}

#[tokio::test]
#[ignore]
async fn test_target_reporters() {
    let ctx = setup().await;

    let target_id = vec![0xAA; 32];
    let target_hex = format!("0x{}", hex::encode(&target_id));
    let reporter_a = vec![0x11; 32];
    let reporter_b = vec![0x22; 32];
    let reporter_c = vec![0x33; 32];
    let reporter_a_hex = format!("0x{}", hex::encode(&reporter_a));
    let reporter_b_hex = format!("0x{}", hex::encode(&reporter_b));
    let reporter_c_hex = format!("0x{}", hex::encode(&reporter_c));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{soma_target_reports, soma_targets};

    // Seed target at cp=5
    diesel::insert_into(soma_targets::table)
        .values(&StoredTarget {
            target_id: target_id.clone(),
            cp_sequence_number: 5,
            epoch: 1,
            status: "Filled".to_string(),
            submitter: Some(vec![0xDD; 32]),
            winning_model_id: None,
            reward_pool: 5000,
            bond_amount: 0,
            report_count: 3,

            winning_distance_score: None,
            winning_loss_score: None,
            winning_model_owner: None,
            fill_epoch: None,
            distance_threshold: 0.5,
            model_ids_json: "[]".to_string(),
            winning_data_url: None,
            winning_data_checksum: None,
            winning_data_size: None,
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Seed 3 reporter rows
    for reporter in [&reporter_a, &reporter_b, &reporter_c] {
        diesel::insert_into(soma_target_reports::table)
            .values(&test_target_report(target_id.clone(), reporter.clone()))
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(r#"{{ target(targetId: "{}") {{ reportCount reporters }} }}"#, target_hex);
    let json = execute(&ctx.schema, &query).await;

    let t = &json["data"]["target"];
    assert_eq!(t["reportCount"], 3);
    let reporters = t["reporters"].as_array().unwrap();
    assert_eq!(reporters.len(), 3);

    let reporter_strs: Vec<&str> = reporters.iter().map(|r| r.as_str().unwrap()).collect();
    assert!(reporter_strs.contains(&reporter_a_hex.as_str()));
    assert!(reporter_strs.contains(&reporter_b_hex.as_str()));
    assert!(reporter_strs.contains(&reporter_c_hex.as_str()));
}

// ---------------------------------------------------------------------------
// Model history
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_model_history() {
    let ctx = setup().await;

    let model_id = vec![0xAA; 32];
    let model_id_hex = format!("0x{}", hex::encode(&model_id));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_models;

    // Model at epochs 1, 3, 5 with increasing stake
    for (epoch, stake) in [(1i64, 100i64), (3, 500), (5, 1200)] {
        let mut m = test_model(model_id.clone(), epoch);
        m.stake = stake;
        diesel::insert_into(soma_models::table).values(&m).execute(conn.deref_mut()).await.unwrap();
    }

    // Full history
    let query = format!(r#"{{ modelHistory(modelId: "{}") {{ epoch stake }} }}"#, model_id_hex);
    let json = execute(&ctx.schema, &query).await;
    let history = json["data"]["modelHistory"].as_array().unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0]["epoch"], "1");
    assert_eq!(history[0]["stake"], "100");
    assert_eq!(history[1]["epoch"], "3");
    assert_eq!(history[1]["stake"], "500");
    assert_eq!(history[2]["epoch"], "5");
    assert_eq!(history[2]["stake"], "1200");

    // With range filter
    let query = format!(
        r#"{{ modelHistory(modelId: "{}", fromEpoch: 2, toEpoch: 4) {{ epoch stake }} }}"#,
        model_id_hex
    );
    let json = execute(&ctx.schema, &query).await;
    let history = json["data"]["modelHistory"].as_array().unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0]["epoch"], "3");
    assert_eq!(history[0]["stake"], "500");
}

// ---------------------------------------------------------------------------
// Model.targets nested resolver (uses winning_model_id)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_model_targets_resolver() {
    let ctx = setup().await;

    let model_id = vec![0xAA; 32];
    let model_id_hex = format!("0x{}", hex::encode(&model_id));
    let target_a = vec![0x11; 32];
    let target_b = vec![0x22; 32];

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{soma_models, soma_targets};

    // Seed model
    diesel::insert_into(soma_models::table)
        .values(&test_model(model_id.clone(), 1))
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Seed 2 targets won by this model
    for (tid, cp) in [(&target_a, 1i64), (&target_b, 2)] {
        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: tid.clone(),
                cp_sequence_number: cp,
                epoch: 1,
                status: "Filled".to_string(),
                submitter: Some(vec![0xDD; 32]),
                winning_model_id: Some(model_id.clone()),
                reward_pool: 1000,
                bond_amount: 0,
                report_count: 0,
                winning_distance_score: Some(0.1),
                winning_loss_score: Some(0.05),
                winning_model_owner: None,
                fill_epoch: Some(1),
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(
        r#"{{ model(modelId: "{}") {{ targets {{ edges {{ node {{ targetId status }} }} }} }} }}"#,
        model_id_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let edges = json["data"]["model"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2);
}

// ---------------------------------------------------------------------------
// Target.reward nested resolver
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_target_reward_resolver() {
    let ctx = setup().await;

    let target_id = vec![0xAA; 32];
    let target_hex = format!("0x{}", hex::encode(&target_id));
    let recipient_a = vec![0x11; 32];
    let recipient_b = vec![0x22; 32];

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{soma_reward_balances, soma_rewards, soma_targets};

    // Seed target
    diesel::insert_into(soma_targets::table)
        .values(&StoredTarget {
            target_id: target_id.clone(),
            cp_sequence_number: 5,
            epoch: 3,
            status: "Claimed".to_string(),
            submitter: Some(vec![0xDD; 32]),
            winning_model_id: Some(vec![0xCC; 32]),
            reward_pool: 5000,
            bond_amount: 100,
            report_count: 0,

            winning_distance_score: Some(0.1),
            winning_loss_score: Some(0.05),
            winning_model_owner: None,
            fill_epoch: Some(3),
            distance_threshold: 0.5,
            model_ids_json: "[]".to_string(),
            winning_data_url: None,
            winning_data_checksum: None,
            winning_data_size: None,
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Seed reward
    diesel::insert_into(soma_rewards::table)
        .values(&StoredReward {
            target_id: target_id.clone(),
            cp_sequence_number: 5,
            epoch: 3,
            tx_digest: vec![0xDD; 32],
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Seed 2 balance rows
    for (recipient, amount) in [(&recipient_a, 1000i64), (&recipient_b, -200)] {
        diesel::insert_into(soma_reward_balances::table)
            .values(&StoredRewardBalance {
                target_id: target_id.clone(),
                cp_sequence_number: 5,
                epoch: 3,
                tx_digest: vec![0xDD; 32],
                recipient: recipient.clone(),
                amount,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(
        r#"{{ target(targetId: "{}") {{ reward {{ epoch balances {{ recipient amount }} }} }} }}"#,
        target_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let reward = &json["data"]["target"]["reward"];
    assert!(!reward.is_null());
    assert_eq!(reward["epoch"], "3");
    let balances = reward["balances"].as_array().unwrap();
    assert_eq!(balances.len(), 2);
}

// ---------------------------------------------------------------------------
// Reward aggregates
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_reward_aggregates() {
    let ctx = setup().await;

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{soma_reward_balances, soma_rewards};

    // 3 rewards at epoch 5
    for i in 0..3u8 {
        let mut tid = vec![0u8; 32];
        tid[0] = i;
        diesel::insert_into(soma_rewards::table)
            .values(&StoredReward {
                target_id: tid.clone(),
                cp_sequence_number: i as i64,
                epoch: 5,
                tx_digest: vec![i; 32],
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // 5 balance rows: amounts [1000, 2000, 500, -100, 3000]
    let amounts = [1000i64, 2000, 500, -100, 3000];
    for (i, amount) in amounts.iter().enumerate() {
        let target_idx = (i % 3) as u8;
        let mut tid = vec![0u8; 32];
        tid[0] = target_idx;
        let mut recipient = vec![0u8; 32];
        recipient[0] = i as u8;
        diesel::insert_into(soma_reward_balances::table)
            .values(&StoredRewardBalance {
                target_id: tid,
                cp_sequence_number: target_idx as i64,
                epoch: 5,
                tx_digest: vec![target_idx; 32],
                recipient,
                amount: *amount,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let json =
        execute(&ctx.schema, r#"{ rewardAggregates(epoch: 5) { totalCount totalAmount } }"#).await;

    let agg = &json["data"]["rewardAggregates"];
    assert_eq!(agg["totalCount"], 3);
    assert_eq!(agg["totalAmount"], "6400");
}

// ---------------------------------------------------------------------------
// Query depth limit
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_query_depth_limit_rejected() {
    let ctx = setup().await;

    // Default max_query_depth is 10. A normal query should succeed.
    let query = "{ serviceConfig { maxPageSize defaultPageSize maxQueryDepth } }";
    let resp = async_graphql::Schema::execute(&ctx.schema, query).await;
    assert!(resp.errors.is_empty(), "normal-depth query should succeed");

    // Build a query exceeding depth 10 via __type introspection nesting.
    let deep_query = r#"{
        __type(name: "Query") {
            fields {
                type {
                    ofType {
                        ofType {
                            ofType {
                                ofType {
                                    ofType {
                                        ofType {
                                            ofType {
                                                name
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }"#;
    let resp = async_graphql::Schema::execute(&ctx.schema, deep_query).await;
    assert!(!resp.errors.is_empty(), "deep query should be rejected by depth limit");
    let err_msg = resp.errors[0].message.to_lowercase();
    assert!(
        err_msg.contains("nested") || err_msg.contains("depth"),
        "error should mention depth limit, got: {err_msg}"
    );
}

// ---------------------------------------------------------------------------
// Query complexity limit
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_query_complexity_limit_rejected() {
    let ctx = setup().await;

    // Default max_query_complexity is 1000.
    // targets(first: 50) has complexity = 5 + 50 * child_complexity.
    // Each Target field is 1, and with ~20 fields + reporters(10) + reward(10) ≈ 40.
    // 5 + 50 * 40 = 2005, which should exceed 1000.
    let query = r#"{
        targets(first: 50) {
            edges {
                node {
                    targetId
                    epoch
                    status
                    submitter
                    winningModelId
                    rewardPool
                    bondAmount
                    reportCount
                    winningDistanceScore
                    winningLossScore
                    winningModelOwner
                    fillEpoch
                    distanceThreshold
                    modelIds
                    winningDataUrl
                    winningDataChecksum
                    winningDataSize
                    reporters
                    reward {
                        epoch
                        balances {
                            recipient
                            amount
                        }
                    }
                }
            }
        }
    }"#;

    let resp = async_graphql::Schema::execute(&ctx.schema, query).await;
    assert!(!resp.errors.is_empty(), "high-complexity query should be rejected");
    let err_msg = resp.errors[0].message.to_lowercase();
    assert!(err_msg.contains("complex"), "error should mention complexity, got: {err_msg}");
}

// ---------------------------------------------------------------------------
// DataLoader batch: reporters across multiple targets
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_targets_batch_reporters() {
    let ctx = setup().await;
    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{soma_target_reports, soma_targets};

    // Seed 3 targets with distinct reporters
    for i in 0..3u8 {
        let mut target_id = vec![0u8; 32];
        target_id[0] = i;

        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: target_id.clone(),
                cp_sequence_number: 5,
                epoch: 1,
                status: "Filled".to_string(),
                submitter: None,
                winning_model_id: None,
                reward_pool: 1000,
                bond_amount: 0,
                report_count: 2,

                winning_distance_score: None,
                winning_loss_score: None,
                winning_model_owner: None,
                fill_epoch: None,
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();

        // Each target gets 2 reporters
        for r in 0..2u8 {
            let mut reporter = vec![0u8; 32];
            reporter[0] = i;
            reporter[1] = r;
            diesel::insert_into(soma_target_reports::table)
                .values(&StoredTargetReport {
                    target_id: target_id.clone(),
                    cp_sequence_number: 5,
                    reporter,
                })
                .execute(conn.deref_mut())
                .await
                .unwrap();
        }
    }

    // Query all 3 targets and their reporters
    let json =
        execute(&ctx.schema, r#"{ targets(first: 3) { edges { node { targetId reporters } } } }"#)
            .await;

    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 3);
    for edge in edges {
        let reporters = edge["node"]["reporters"].as_array().unwrap();
        assert_eq!(reporters.len(), 2, "each target should have 2 reporters");
    }
}

// ---------------------------------------------------------------------------
// DataLoader batch: rewards across multiple targets
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_targets_batch_rewards() {
    let ctx = setup().await;
    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{soma_reward_balances, soma_rewards, soma_targets};

    // Seed 3 claimed targets with rewards and balances
    for i in 0..3u8 {
        let mut target_id = vec![0u8; 32];
        target_id[0] = i;
        let mut tx_digest = vec![0u8; 32];
        tx_digest[0] = i;

        diesel::insert_into(soma_targets::table)
            .values(&StoredTarget {
                target_id: target_id.clone(),
                cp_sequence_number: 5,
                epoch: 1,
                status: "Claimed".to_string(),
                submitter: Some(vec![0xDD; 32]),
                winning_model_id: Some(vec![0xCC; 32]),
                reward_pool: 5000,
                bond_amount: 100,
                report_count: 0,

                winning_distance_score: Some(0.1),
                winning_loss_score: Some(0.05),
                winning_model_owner: None,
                fill_epoch: Some(1),
                distance_threshold: 0.5,
                model_ids_json: "[]".to_string(),
                winning_data_url: None,
                winning_data_checksum: None,
                winning_data_size: None,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();

        diesel::insert_into(soma_rewards::table)
            .values(&StoredReward {
                target_id: target_id.clone(),
                cp_sequence_number: 5,
                epoch: 1,
                tx_digest: tx_digest.clone(),
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();

        // Each reward has 1 balance row
        let mut recipient = vec![0u8; 32];
        recipient[0] = i;
        diesel::insert_into(soma_reward_balances::table)
            .values(&StoredRewardBalance {
                target_id: target_id.clone(),
                cp_sequence_number: 5,
                epoch: 1,
                tx_digest,
                recipient,
                amount: (i as i64 + 1) * 1000,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // Query all 3 targets with nested rewards
    let json = execute(
        &ctx.schema,
        r#"{ targets(first: 3) { edges { node { targetId reward { epoch balances { recipient amount } } } } } }"#,
    )
    .await;

    let edges = json["data"]["targets"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 3);
    for edge in edges {
        let reward = &edge["node"]["reward"];
        assert!(!reward.is_null(), "each target should have a reward");
        assert_eq!(reward["epoch"], "1");
        let balances = reward["balances"].as_array().unwrap();
        assert_eq!(balances.len(), 1, "each reward should have 1 balance");
    }
}

#[tokio::test]
#[ignore]
async fn test_model_leaderboard() {
    let ctx = setup().await;

    let model_a = vec![0x11; 32];
    let model_b = vec![0x22; 32];
    let model_a_hex = format!("0x{}", hex::encode(&model_a));
    let model_b_hex = format!("0x{}", hex::encode(&model_b));
    let model_ids_json = serde_json::to_string(&vec![&model_a_hex, &model_b_hex]).unwrap();

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_targets;

    // Target 1: filled, won by model_a, both models assigned
    diesel::insert_into(soma_targets::table)
        .values(&StoredTarget {
            target_id: vec![0x01; 32],
            cp_sequence_number: 10,
            epoch: 1,
            status: "Filled".to_string(),
            submitter: Some(vec![0xAA; 32]),
            winning_model_id: Some(model_a.clone()),
            reward_pool: 5000,
            bond_amount: 0,
            report_count: 0,
            winning_distance_score: Some(0.1),
            winning_loss_score: Some(0.05),
            winning_model_owner: Some(vec![0xCC; 32]),
            fill_epoch: Some(1),
            distance_threshold: 0.5,
            model_ids_json: model_ids_json.clone(),
            winning_data_url: None,
            winning_data_checksum: None,
            winning_data_size: Some(1024),
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Target 2: filled, won by model_a, both models assigned
    diesel::insert_into(soma_targets::table)
        .values(&StoredTarget {
            target_id: vec![0x02; 32],
            cp_sequence_number: 11,
            epoch: 1,
            status: "Filled".to_string(),
            submitter: Some(vec![0xAA; 32]),
            winning_model_id: Some(model_a.clone()),
            reward_pool: 3000,
            bond_amount: 0,
            report_count: 0,
            winning_distance_score: Some(0.2),
            winning_loss_score: Some(0.03),
            winning_model_owner: Some(vec![0xCC; 32]),
            fill_epoch: Some(1),
            distance_threshold: 0.5,
            model_ids_json: model_ids_json.clone(),
            winning_data_url: None,
            winning_data_checksum: None,
            winning_data_size: Some(2048),
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Target 3: filled, won by model_b, both models assigned
    diesel::insert_into(soma_targets::table)
        .values(&StoredTarget {
            target_id: vec![0x03; 32],
            cp_sequence_number: 12,
            epoch: 1,
            status: "Filled".to_string(),
            submitter: Some(vec![0xAA; 32]),
            winning_model_id: Some(model_b.clone()),
            reward_pool: 4000,
            bond_amount: 0,
            report_count: 0,
            winning_distance_score: Some(0.15),
            winning_loss_score: Some(0.04),
            winning_model_owner: Some(vec![0xDD; 32]),
            fill_epoch: Some(1),
            distance_threshold: 0.5,
            model_ids_json: model_ids_json.clone(),
            winning_data_url: None,
            winning_data_checksum: None,
            winning_data_size: Some(512),
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Target 4: open (not filled), both models assigned — counts toward assignments only
    diesel::insert_into(soma_targets::table)
        .values(&StoredTarget {
            target_id: vec![0x04; 32],
            cp_sequence_number: 13,
            epoch: 1,
            status: "Open".to_string(),
            submitter: None,
            winning_model_id: None,
            reward_pool: 2000,
            bond_amount: 0,
            report_count: 0,
            winning_distance_score: None,
            winning_loss_score: None,
            winning_model_owner: None,
            fill_epoch: None,
            distance_threshold: 0.5,
            model_ids_json: model_ids_json.clone(),
            winning_data_url: None,
            winning_data_checksum: None,
            winning_data_size: None,
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    // Query the model leaderboard
    let query = r#"{ modelLeaderboard { edges { node { modelId targetsWon targetsAssigned winRate avgDistanceScore avgLossScore totalReward totalDataSize } } } }"#;
    let json = execute(&ctx.schema, query).await;

    let edges = json["data"]["modelLeaderboard"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 2, "should have 2 models");

    // model_a should be first (2 wins), model_b second (1 win)
    let a = &edges[0]["node"];
    assert_eq!(a["modelId"], model_a_hex);
    assert_eq!(a["targetsWon"], 2);
    assert_eq!(a["targetsAssigned"], 4); // assigned in all 4 targets
    // win rate = 2/4 = 0.5
    let win_rate_a: f64 = a["winRate"].as_f64().unwrap();
    assert!((win_rate_a - 0.5).abs() < 1e-6);
    assert_eq!(a["totalReward"], "8000"); // 5000 + 3000

    let b = &edges[1]["node"];
    assert_eq!(b["modelId"], model_b_hex);
    assert_eq!(b["targetsWon"], 1);
    assert_eq!(b["targetsAssigned"], 4);
    let win_rate_b: f64 = b["winRate"].as_f64().unwrap();
    assert!((win_rate_b - 0.25).abs() < 1e-6);
    assert_eq!(b["totalReward"], "4000");
}
