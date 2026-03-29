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
use indexer_alt_schema::soma::{StoredEpochState, StoredStakedSoma};
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
// Marketplace: asks
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_asks_query() {
    let ctx = setup().await;

    let buyer = vec![0x11; 32];
    let buyer_hex = format!("0x{}", hex::encode(&buyer));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_asks;
    use indexer_alt_schema::soma::StoredAsk;

    for i in 0..3u8 {
        let mut ask_id = vec![0u8; 32];
        ask_id[0] = i;
        diesel::insert_into(soma_asks::table)
            .values(&StoredAsk {
                ask_id,
                cp_sequence_number: i as i64,
                buyer: buyer.clone(),
                task_digest: vec![i; 32],
                max_price_per_bid: 1000,
                num_bids_wanted: 1,
                timeout_ms: 60_000,
                created_at_ms: 1_000_000 + i as i64,
                status: "open".to_string(),
                accepted_bid_count: 0,
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    let query = format!(
        r#"{{ asks(buyer: "{}") {{ askId buyer status maxPricePerBid numBidsWanted }} }}"#,
        buyer_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let asks = json["data"]["asks"].as_array().unwrap();
    assert_eq!(asks.len(), 3);
    for a in asks {
        assert_eq!(a["buyer"], buyer_hex);
        assert_eq!(a["status"], "open");
        assert_eq!(a["maxPricePerBid"], "1000");
    }
}

// ---------------------------------------------------------------------------
// Marketplace: settlements
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_settlements_query() {
    let ctx = setup().await;

    let buyer = vec![0x11; 32];
    let seller = vec![0x22; 32];
    let seller_hex = format!("0x{}", hex::encode(&seller));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::soma_settlements;
    use indexer_alt_schema::soma::StoredSettlement;

    let mut sid = vec![0u8; 32];
    sid[0] = 1;
    diesel::insert_into(soma_settlements::table)
        .values(&StoredSettlement {
            settlement_id: sid,
            cp_sequence_number: 1,
            ask_id: vec![0xAA; 32],
            bid_id: vec![0xBB; 32],
            buyer: buyer.clone(),
            seller: seller.clone(),
            amount: 950,
            task_digest: vec![0xCC; 32],
            response_digest: vec![0xDD; 32],
            settled_at_ms: 2_000_000,
            seller_rating: "positive".to_string(),
            rating_deadline_ms: 3_000_000,
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let query = format!(
        r#"{{ settlements(seller: "{}") {{ seller amount sellerRating }} }}"#,
        seller_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let settlements = json["data"]["settlements"].as_array().unwrap();
    assert_eq!(settlements.len(), 1);
    assert_eq!(settlements[0]["seller"], seller_hex);
    assert_eq!(settlements[0]["amount"], "950");
    assert_eq!(settlements[0]["sellerRating"], "positive");
}

// ---------------------------------------------------------------------------
// Marketplace: reputation
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_reputation_query() {
    let ctx = setup().await;

    let seller = vec![0x22; 32];
    let seller_hex = format!("0x{}", hex::encode(&seller));

    let mut conn = ctx.db.connect().await.unwrap();
    use indexer_alt_schema::schema::{soma_bids, soma_settlements};
    use indexer_alt_schema::soma::{StoredBid, StoredSettlement};

    // Insert 2 bids from seller (1 accepted, 1 pending)
    for (i, status) in [(0u8, "accepted"), (1, "pending")] {
        let mut bid_id = vec![0u8; 32];
        bid_id[0] = i;
        diesel::insert_into(soma_bids::table)
            .values(&StoredBid {
                bid_id,
                cp_sequence_number: i as i64,
                ask_id: vec![0xAA; 32],
                seller: seller.clone(),
                price: 1000,
                response_digest: vec![i; 32],
                created_at_ms: 1_000_000,
                status: status.to_string(),
            })
            .execute(conn.deref_mut())
            .await
            .unwrap();
    }

    // Insert 1 settlement (positive rating)
    let mut sid = vec![0u8; 32];
    sid[0] = 1;
    diesel::insert_into(soma_settlements::table)
        .values(&StoredSettlement {
            settlement_id: sid,
            cp_sequence_number: 1,
            ask_id: vec![0xAA; 32],
            bid_id: vec![0u8; 32],
            buyer: vec![0x11; 32],
            seller: seller.clone(),
            amount: 950,
            task_digest: vec![0xCC; 32],
            response_digest: vec![0xDD; 32],
            settled_at_ms: 2_000_000,
            seller_rating: "positive".to_string(),
            rating_deadline_ms: 3_000_000,
        })
        .execute(conn.deref_mut())
        .await
        .unwrap();

    let query = format!(
        r#"{{ reputation(address: "{}") {{ totalBidsSubmitted totalBidsWon sellerApprovalRate bidToWinRatio totalVolumeEarned }} }}"#,
        seller_hex
    );
    let json = execute(&ctx.schema, &query).await;

    let rep = &json["data"]["reputation"];
    assert_eq!(rep["totalBidsSubmitted"], 2);
    assert_eq!(rep["totalBidsWon"], 1);
    assert_eq!(rep["sellerApprovalRate"], 1.0); // 1 positive, 0 negative
    assert_eq!(rep["bidToWinRatio"], 0.5); // 1 won / 2 submitted
    assert_eq!(rep["totalVolumeEarned"], "950");
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
        let obj = Object::new_coin(object_id, types::object::CoinType::Soma, balance, owner, TransactionDigest::ZERO);
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
        let obj = Object::new_coin(object_id, types::object::CoinType::Soma, 5000, owner, TransactionDigest::ZERO);
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

    // A's balance must be 0 -- coin was transferred away
    let query_a = format!(r#"{{ balance(address: "{}") }}"#, addr_a_hex);
    let json_a = execute(&ctx.schema, &query_a).await;
    assert_eq!(json_a["data"]["balance"], "0");

    // B's balance must be 5000
    let query_b = format!(r#"{{ balance(address: "{}") }}"#, addr_b_hex);
    let json_b = execute(&ctx.schema, &query_b).await;
    assert_eq!(json_b["data"]["balance"], "5000");
}

// ---------------------------------------------------------------------------
// Epoch state
// ---------------------------------------------------------------------------

fn test_epoch_state(epoch: i64) -> StoredEpochState {
    StoredEpochState {
        epoch,
        emission_balance: 0,
        emission_per_epoch: 0,
        distribution_counter: 0,
        period_length: 10,
        decrease_rate: 1000,
        protocol_fund_balance: 0,
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
    es5.distribution_counter = 5;
    es5.protocol_fund_balance = 25_000;
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
        r#"{ epochState(epoch: 5) { epoch emissionBalance emissionPerEpoch distributionCounter periodLength decreaseRate protocolFundBalance safeMode safeModeAccumulatedFees safeModeAccumulatedEmissions } }"#,
    )
    .await;

    let es = &json["data"]["epochState"];
    assert_eq!(es["epoch"], "5");
    assert_eq!(es["emissionBalance"], "1000000");
    assert_eq!(es["emissionPerEpoch"], "10000");
    assert_eq!(es["distributionCounter"], "5");
    assert_eq!(es["periodLength"], "10");
    assert_eq!(es["decreaseRate"], 1000);
    assert_eq!(es["protocolFundBalance"], "25000");
    assert_eq!(es["safeMode"], false);
    assert_eq!(es["safeModeAccumulatedFees"], "0");
    assert_eq!(es["safeModeAccumulatedEmissions"], "0");

    // Query latest (no arg) -> epoch 7
    let json = execute(&ctx.schema, r#"{ epochState { epoch } }"#).await;
    assert_eq!(json["data"]["epochState"]["epoch"], "7");
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
