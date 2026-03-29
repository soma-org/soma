// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::ops::DerefMut;
use std::sync::Arc;

use async_graphql::connection::{Connection, Edge};
use async_graphql::{Context, Error, Object, Result};
use diesel::ExpressionMethods;
use diesel::OptionalExtension;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;

use crate::api::scalars::BigInt;
use crate::api::types::address::Address;
use crate::api::types::ask::Ask;
use crate::api::types::available_range::AvailableRange;
use crate::api::types::bid::Bid;
use crate::api::types::checkpoint::Checkpoint;
use crate::api::types::epoch::Epoch;
use crate::api::types::epoch_state::EpochState;
use crate::api::types::network_metrics::NetworkMetrics;
use crate::api::types::object::Object as GqlObject;
use crate::api::types::reputation::Reputation;
use crate::api::types::service_config::ServiceConfig;
use crate::api::types::settlement::Settlement;
use crate::api::types::staked_soma::StakedSoma;
use crate::api::types::transaction::Transaction;
use crate::api::types::transaction_detail::TransactionDetail;
use crate::api::types::validator::Validator;
use crate::config::GraphQlConfig;
use crate::db::PgReader;

use indexer_kvstore::KvLoader;

/// Try to get the KvLoader from the GraphQL context. Returns None if not configured.
fn kv_loader<'a>(ctx: &'a Context<'a>) -> Option<&'a Arc<dyn KvLoader>> {
    ctx.data::<Arc<dyn KvLoader>>().ok()
}

pub struct Query;

#[Object]
impl Query {
    /// The chain identifier (base58 hash of genesis checkpoint summary).
    async fn chain_identifier(&self, ctx: &Context<'_>) -> Result<String> {
        // BigTable path: read genesis checkpoint from KvLoader
        if let Some(kv) = kv_loader(ctx) {
            if let Some(cp) = kv.get_checkpoint(0).await.map_err(|e| Error::new(e.to_string()))? {
                let bytes = bcs::to_bytes(&cp.summary).map_err(|e| Error::new(e.to_string()))?;
                return Ok(bs58::encode(&bytes).into_string());
            }
        }

        // Postgres fallback
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::kv_checkpoints;
        let summary: Option<Vec<u8>> = kv_checkpoints::table
            .select(kv_checkpoints::checkpoint_summary)
            .filter(kv_checkpoints::sequence_number.eq(0i64))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        match summary {
            Some(bytes) => Ok(bs58::encode(&bytes).into_string()),
            None => Ok("unknown".to_string()),
        }
    }

    /// Fetch the service configuration.
    async fn service_config(&self, ctx: &Context<'_>) -> Result<ServiceConfig> {
        let config: &GraphQlConfig = ctx.data()?;
        Ok(ServiceConfig {
            max_page_size: config.max_page_size,
            default_page_size: config.default_page_size,
            max_query_depth: config.max_query_depth as i32,
        })
    }

    /// Look up a checkpoint by sequence number. Returns the latest if omitted.
    async fn checkpoint(
        &self,
        ctx: &Context<'_>,
        sequence_number: Option<i64>,
    ) -> Result<Option<Checkpoint>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::{cp_sequence_numbers, kv_checkpoints};

        // "Latest" discovery always uses cp_sequence_numbers (never pruned).
        let seq = match sequence_number {
            Some(s) => s,
            None => cp_sequence_numbers::table
                .select(cp_sequence_numbers::cp_sequence_number)
                .order(cp_sequence_numbers::cp_sequence_number.desc())
                .first::<i64>(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?
                .ok_or_else(|| Error::new("No checkpoints found"))?,
        };

        // Metadata from cp_sequence_numbers (never pruned).
        let cp_info: Option<(i64, i64)> = cp_sequence_numbers::table
            .select((cp_sequence_numbers::epoch, cp_sequence_numbers::tx_lo))
            .filter(cp_sequence_numbers::cp_sequence_number.eq(seq))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        // BigTable path: read BCS content from KvLoader
        if let Some(kv) = kv_loader(ctx) {
            let cp = kv.get_checkpoint(seq as u64).await.map_err(|e| Error::new(e.to_string()))?;

            let Some(cp) = cp else {
                return Ok(None);
            };

            return Ok(Some(Checkpoint {
                sequence_number: seq,
                checkpoint_summary_bcs: bcs::to_bytes(&cp.summary)
                    .map_err(|e| Error::new(e.to_string()))?,
                checkpoint_contents_bcs: bcs::to_bytes(&cp.contents)
                    .map_err(|e| Error::new(e.to_string()))?,
                validator_signatures_bcs: bcs::to_bytes(&cp.signatures)
                    .map_err(|e| Error::new(e.to_string()))?,
                epoch: cp_info.map(|(e, _)| e),
                tx_lo: cp_info.map(|(_, t)| t),
                timestamp_ms: None,
            }));
        }

        // Postgres fallback
        let stored: Option<indexer_alt_schema::checkpoints::StoredCheckpoint> =
            kv_checkpoints::table
                .filter(kv_checkpoints::sequence_number.eq(seq))
                .first(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?;

        let Some(stored) = stored else {
            return Ok(None);
        };

        Ok(Some(Checkpoint {
            sequence_number: stored.sequence_number,
            checkpoint_summary_bcs: stored.checkpoint_summary,
            checkpoint_contents_bcs: stored.checkpoint_contents,
            validator_signatures_bcs: stored.validator_signatures,
            epoch: cp_info.map(|(e, _)| e),
            tx_lo: cp_info.map(|(_, t)| t),
            timestamp_ms: None,
        }))
    }

    /// Look up a transaction by digest (base58).
    async fn transaction(&self, ctx: &Context<'_>, digest: String) -> Result<Option<Transaction>> {
        let digest_bytes = bs58::decode(&digest)
            .into_vec()
            .map_err(|e| Error::new(format!("Invalid digest: {e}")))?;

        // BigTable path
        if let Some(kv) = kv_loader(ctx) {
            let arr: [u8; 32] = digest_bytes
                .as_slice()
                .try_into()
                .map_err(|_| Error::new("Digest must be 32 bytes"))?;
            let tx_digest = types::digests::TransactionDigest::new(arr);
            let tx = kv.get_transaction(&tx_digest).await.map_err(|e| Error::new(e.to_string()))?;

            return Ok(tx.map(|t| Transaction {
                tx_digest: digest_bytes,
                cp_sequence_number: t.checkpoint_number as i64,
                timestamp_ms: t.timestamp as i64,
                raw_transaction_bcs: bcs::to_bytes(&t.transaction).unwrap_or_default(),
                raw_effects_bcs: bcs::to_bytes(&t.effects).unwrap_or_default(),
                user_signatures_bcs: bcs::to_bytes(t.transaction.tx_signatures())
                    .unwrap_or_default(),
            }));
        }

        // Postgres fallback
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::kv_transactions;
        let stored: Option<indexer_alt_schema::transactions::StoredTransaction> =
            kv_transactions::table
                .filter(kv_transactions::tx_digest.eq(&digest_bytes))
                .first(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?;

        Ok(stored.map(|s| Transaction {
            tx_digest: s.tx_digest,
            cp_sequence_number: s.cp_sequence_number,
            timestamp_ms: s.timestamp_ms,
            raw_transaction_bcs: s.raw_transaction,
            raw_effects_bcs: s.raw_effects,
            user_signatures_bcs: s.user_signatures,
        }))
    }

    /// Look up decoded transaction detail by digest (base58).
    async fn transaction_detail(
        &self,
        ctx: &Context<'_>,
        digest: String,
    ) -> Result<Option<TransactionDetail>> {
        let digest_bytes = bs58::decode(&digest)
            .into_vec()
            .map_err(|e| Error::new(format!("Invalid digest: {e}")))?;

        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_tx_details;

        type Row = (i64, Vec<u8>, String, Vec<u8>, i64, i64, Option<String>);

        let stored: Option<Row> = soma_tx_details::table
            .select((
                soma_tx_details::tx_sequence_number,
                soma_tx_details::tx_digest,
                soma_tx_details::kind,
                soma_tx_details::sender,
                soma_tx_details::epoch,
                soma_tx_details::timestamp_ms,
                soma_tx_details::metadata_json,
            ))
            .filter(soma_tx_details::tx_digest.eq(&digest_bytes))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(stored.map(|row| TransactionDetail {
            tx_sequence_number: row.0,
            tx_digest: row.1,
            kind: row.2,
            sender: row.3,
            epoch: row.4,
            timestamp_ms: row.5,
            metadata_json: row.6,
        }))
    }

    /// Look up an object by ID (hex with 0x prefix) and optional version.
    async fn object(
        &self,
        ctx: &Context<'_>,
        id: String,
        version: Option<i64>,
    ) -> Result<Option<GqlObject>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let id_hex = id.strip_prefix("0x").unwrap_or(&id);
        let id_bytes =
            hex::decode(id_hex).map_err(|e| Error::new(format!("Invalid object ID: {e}")))?;

        use indexer_alt_schema::schema::{kv_objects, obj_info, obj_versions};

        // Version discovery from obj_versions (Tier C, never pruned)
        let ver = match version {
            Some(v) => v,
            None => obj_versions::table
                .select(obj_versions::object_version)
                .filter(obj_versions::object_id.eq(&id_bytes))
                .order(obj_versions::object_version.desc())
                .first::<i64>(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?
                .ok_or_else(|| Error::new("Object not found"))?,
        };

        // Metadata from obj_info (Tier C, never pruned)
        let info: Option<indexer_alt_schema::objects::StoredObjInfo> = obj_info::table
            .filter(obj_info::object_id.eq(&id_bytes))
            .order(obj_info::cp_sequence_number.desc())
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        let owner_kind = info.as_ref().and_then(|i| {
            i.owner_kind.map(|k| match k {
                indexer_alt_schema::objects::StoredOwnerKind::Immutable => "Immutable",
                indexer_alt_schema::objects::StoredOwnerKind::Address => "Address",
                indexer_alt_schema::objects::StoredOwnerKind::Object => "Object",
                indexer_alt_schema::objects::StoredOwnerKind::Shared => "Shared",
            })
        });

        let object_type = info.as_ref().and_then(|i| i.module.as_ref().cloned());

        // BCS content: BigTable path
        if let Some(kv) = kv_loader(ctx) {
            let obj_id = types::object::ObjectID::from_bytes(&id_bytes)
                .map_err(|e| Error::new(e.to_string()))?;
            let obj =
                kv.get_object(&obj_id, ver as u64).await.map_err(|e| Error::new(e.to_string()))?;

            return Ok(Some(GqlObject {
                object_id: id_bytes,
                object_version: ver,
                serialized_object_bcs: obj.map(|o| bcs::to_bytes(&o).unwrap_or_default()),
                owner_kind: owner_kind.map(String::from),
                owner_id: info.and_then(|i| i.owner_id),
                object_type,
            }));
        }

        // Postgres fallback for BCS content
        let stored: Option<indexer_alt_schema::objects::StoredObject> = kv_objects::table
            .filter(kv_objects::object_id.eq(&id_bytes))
            .filter(kv_objects::object_version.eq(ver))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        let Some(stored) = stored else {
            return Ok(None);
        };

        Ok(Some(GqlObject {
            object_id: stored.object_id,
            object_version: stored.object_version,
            serialized_object_bcs: stored.serialized_object,
            owner_kind: owner_kind.map(String::from),
            owner_id: info.and_then(|i| i.owner_id),
            object_type,
        }))
    }

    /// Look up an address to query its transactions.
    async fn address(&self, address: String) -> Result<Address> {
        let addr_hex = address.strip_prefix("0x").unwrap_or(&address);
        let addr_bytes =
            hex::decode(addr_hex).map_err(|e| Error::new(format!("Invalid address: {e}")))?;
        Ok(Address { address: addr_bytes })
    }

    /// Look up an epoch by ID. Returns the latest if omitted.
    async fn epoch(&self, ctx: &Context<'_>, epoch_id: Option<i64>) -> Result<Option<Epoch>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::{cp_sequence_numbers, kv_epoch_ends, kv_epoch_starts};

        // "Latest" epoch discovery uses cp_sequence_numbers (never pruned).
        let epoch_num = match epoch_id {
            Some(e) => e,
            None => cp_sequence_numbers::table
                .select(cp_sequence_numbers::epoch)
                .order(cp_sequence_numbers::epoch.desc())
                .first::<i64>(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?
                .ok_or_else(|| Error::new("No epochs found"))?,
        };

        // BigTable path
        if let Some(kv) = kv_loader(ctx) {
            let epoch_data =
                kv.get_epoch(epoch_num as u64).await.map_err(|e| Error::new(e.to_string()))?;

            let Some(ed) = epoch_data else {
                return Ok(None);
            };

            return Ok(Some(Epoch {
                epoch: ed.epoch.unwrap_or(epoch_num as u64) as i64,
                protocol_version: ed.protocol_version.unwrap_or(0) as i64,
                cp_lo: ed.start_checkpoint.unwrap_or(0) as i64,
                start_timestamp_ms: ed.start_timestamp_ms.unwrap_or(0) as i64,
                reference_gas_price: ed.reference_gas_price.unwrap_or(0) as i64,
                system_state_bcs: ed.system_state_bcs.unwrap_or_default(),
                cp_hi: ed.cp_hi.map(|v| v as i64),
                tx_hi: ed.tx_hi.map(|v| v as i64),
                end_timestamp_ms: ed.end_timestamp_ms.map(|v| v as i64),
                safe_mode: ed.safe_mode,
                total_stake: None,
                total_gas_fees: None,
            }));
        }

        // Postgres fallback
        let start: Option<indexer_alt_schema::epochs::StoredEpochStart> = kv_epoch_starts::table
            .filter(kv_epoch_starts::epoch.eq(epoch_num))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        let Some(start) = start else {
            return Ok(None);
        };

        let end: Option<indexer_alt_schema::epochs::StoredEpochEnd> = kv_epoch_ends::table
            .filter(kv_epoch_ends::epoch.eq(epoch_num))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(Some(Epoch {
            epoch: start.epoch,
            protocol_version: start.protocol_version,
            cp_lo: start.cp_lo,
            start_timestamp_ms: start.start_timestamp_ms,
            reference_gas_price: start.reference_gas_price,
            system_state_bcs: start.system_state,
            cp_hi: end.as_ref().map(|e| e.cp_hi),
            tx_hi: end.as_ref().map(|e| e.tx_hi),
            end_timestamp_ms: end.as_ref().map(|e| e.end_timestamp_ms),
            safe_mode: end.as_ref().map(|e| e.safe_mode),
            total_stake: end.as_ref().and_then(|e| e.total_stake),
            total_gas_fees: end.as_ref().and_then(|e| e.total_gas_fees),
        }))
    }

    /// Query marketplace asks with optional filters.
    async fn asks(
        &self,
        ctx: &Context<'_>,
        status: Option<String>,
        buyer: Option<String>,
        limit: Option<i64>,
    ) -> Result<Vec<Ask>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_asks;

        type AskRow = (Vec<u8>, Vec<u8>, Vec<u8>, i64, i32, i64, i64, String, i32);

        let mut query = soma_asks::table
            .select((
                soma_asks::ask_id,
                soma_asks::buyer,
                soma_asks::task_digest,
                soma_asks::max_price_per_bid,
                soma_asks::num_bids_wanted,
                soma_asks::timeout_ms,
                soma_asks::created_at_ms,
                soma_asks::status,
                soma_asks::accepted_bid_count,
            ))
            .order(soma_asks::created_at_ms.desc())
            .limit(limit.unwrap_or(50))
            .into_boxed();

        if let Some(ref s) = status {
            query = query.filter(soma_asks::status.eq(s));
        }

        if let Some(ref b) = buyer {
            let hex = b.strip_prefix("0x").unwrap_or(b);
            let bytes =
                hex::decode(hex).map_err(|e| Error::new(format!("Invalid buyer address: {e}")))?;
            query = query.filter(soma_asks::buyer.eq(bytes));
        }

        let rows: Vec<AskRow> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| Ask {
                ask_id: r.0,
                buyer: r.1,
                task_digest: r.2,
                max_price_per_bid: r.3,
                num_bids_wanted: r.4,
                timeout_ms: r.5,
                created_at_ms: r.6,
                status: r.7,
                accepted_bid_count: r.8,
            })
            .collect())
    }

    /// Query bids for a specific ask.
    async fn bids(
        &self,
        ctx: &Context<'_>,
        ask_id: Option<String>,
        seller: Option<String>,
        status: Option<String>,
        limit: Option<i64>,
    ) -> Result<Vec<Bid>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_bids;

        type BidRow = (Vec<u8>, Vec<u8>, Vec<u8>, i64, Vec<u8>, i64, String);

        let mut query = soma_bids::table
            .select((
                soma_bids::bid_id,
                soma_bids::ask_id,
                soma_bids::seller,
                soma_bids::price,
                soma_bids::response_digest,
                soma_bids::created_at_ms,
                soma_bids::status,
            ))
            .order(soma_bids::created_at_ms.desc())
            .limit(limit.unwrap_or(50))
            .into_boxed();

        if let Some(ref a) = ask_id {
            let hex = a.strip_prefix("0x").unwrap_or(a);
            let bytes =
                hex::decode(hex).map_err(|e| Error::new(format!("Invalid ask ID: {e}")))?;
            query = query.filter(soma_bids::ask_id.eq(bytes));
        }

        if let Some(ref s) = seller {
            let hex = s.strip_prefix("0x").unwrap_or(s);
            let bytes = hex::decode(hex)
                .map_err(|e| Error::new(format!("Invalid seller address: {e}")))?;
            query = query.filter(soma_bids::seller.eq(bytes));
        }

        if let Some(ref st) = status {
            query = query.filter(soma_bids::status.eq(st));
        }

        let rows: Vec<BidRow> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| Bid {
                bid_id: r.0,
                ask_id: r.1,
                seller: r.2,
                price: r.3,
                response_digest: r.4,
                created_at_ms: r.5,
                status: r.6,
            })
            .collect())
    }

    /// Query settlements with optional filters.
    async fn settlements(
        &self,
        ctx: &Context<'_>,
        buyer: Option<String>,
        seller: Option<String>,
        limit: Option<i64>,
    ) -> Result<Vec<Settlement>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_settlements;

        type Row = (
            Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, i64,
            Vec<u8>, Vec<u8>, i64, String, i64,
        );

        let mut query = soma_settlements::table
            .select((
                soma_settlements::settlement_id,
                soma_settlements::ask_id,
                soma_settlements::bid_id,
                soma_settlements::buyer,
                soma_settlements::seller,
                soma_settlements::amount,
                soma_settlements::task_digest,
                soma_settlements::response_digest,
                soma_settlements::settled_at_ms,
                soma_settlements::seller_rating,
                soma_settlements::rating_deadline_ms,
            ))
            .order(soma_settlements::settled_at_ms.desc())
            .limit(limit.unwrap_or(50))
            .into_boxed();

        if let Some(ref b) = buyer {
            let hex = b.strip_prefix("0x").unwrap_or(b);
            let bytes =
                hex::decode(hex).map_err(|e| Error::new(format!("Invalid buyer address: {e}")))?;
            query = query.filter(soma_settlements::buyer.eq(bytes));
        }

        if let Some(ref s) = seller {
            let hex = s.strip_prefix("0x").unwrap_or(s);
            let bytes = hex::decode(hex)
                .map_err(|e| Error::new(format!("Invalid seller address: {e}")))?;
            query = query.filter(soma_settlements::seller.eq(bytes));
        }

        let rows: Vec<Row> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|r| Settlement {
                settlement_id: r.0,
                ask_id: r.1,
                bid_id: r.2,
                buyer: r.3,
                seller: r.4,
                amount: r.5,
                task_digest: r.6,
                response_digest: r.7,
                settled_at_ms: r.8,
                seller_rating: r.9,
                rating_deadline_ms: r.10,
            })
            .collect())
    }

    /// Query reputation for an address, computed from settlement history.
    async fn reputation(
        &self,
        ctx: &Context<'_>,
        address: String,
    ) -> Result<Reputation> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let hex = address.strip_prefix("0x").unwrap_or(&address);
        let addr_bytes =
            hex::decode(hex).map_err(|e| Error::new(format!("Invalid address: {e}")))?;

        use indexer_alt_schema::schema::{soma_asks, soma_bids, soma_settlements};

        // Buyer metrics
        let total_asks_created: i64 = soma_asks::table
            .filter(soma_asks::buyer.eq(&addr_bytes))
            .count()
            .get_result(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let total_bids_accepted: i64 = soma_settlements::table
            .filter(soma_settlements::buyer.eq(&addr_bytes))
            .count()
            .get_result(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let total_volume_spent: Option<i64> = soma_settlements::table
            .filter(soma_settlements::buyer.eq(&addr_bytes))
            .select(diesel::dsl::sql::<diesel::sql_types::Nullable<diesel::sql_types::BigInt>>(
                "COALESCE(SUM(amount), 0)::BIGINT",
            ))
            .first(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let unique_sellers: i64 = soma_settlements::table
            .filter(soma_settlements::buyer.eq(&addr_bytes))
            .select(diesel::dsl::sql::<diesel::sql_types::BigInt>(
                "COUNT(DISTINCT seller)",
            ))
            .first(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Seller metrics
        let total_bids_submitted: i64 = soma_bids::table
            .filter(soma_bids::seller.eq(&addr_bytes))
            .count()
            .get_result(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let total_settlements_as_seller: i64 = soma_settlements::table
            .filter(soma_settlements::seller.eq(&addr_bytes))
            .count()
            .get_result(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let total_volume_earned: Option<i64> = soma_settlements::table
            .filter(soma_settlements::seller.eq(&addr_bytes))
            .select(diesel::dsl::sql::<diesel::sql_types::Nullable<diesel::sql_types::BigInt>>(
                "COALESCE(SUM(amount), 0)::BIGINT",
            ))
            .first(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let negative_ratings_received: i64 = soma_settlements::table
            .filter(soma_settlements::seller.eq(&addr_bytes))
            .filter(soma_settlements::seller_rating.eq("negative"))
            .count()
            .get_result(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let total_bids_won: i64 = soma_bids::table
            .filter(soma_bids::seller.eq(&addr_bytes))
            .filter(soma_bids::status.eq("accepted"))
            .count()
            .get_result(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(Reputation {
            address: addr_bytes,
            total_asks_created,
            total_bids_accepted,
            total_volume_spent: total_volume_spent.unwrap_or(0),
            unique_sellers,
            total_bids_submitted,
            total_bids_won,
            total_volume_earned: total_volume_earned.unwrap_or(0),
            negative_ratings_received,
            total_settlements_as_seller,
        })
    }

    /// Look up a staked SOMA position by its object ID.
    async fn staked_soma(&self, ctx: &Context<'_>, id: String) -> Result<Option<StakedSoma>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let id_hex = id.strip_prefix("0x").unwrap_or(&id);
        let id_bytes =
            hex::decode(id_hex).map_err(|e| Error::new(format!("Invalid staked soma ID: {e}")))?;

        use indexer_alt_schema::schema::soma_staked_soma;

        type Row = (Vec<u8>, i64, Option<Vec<u8>>, Option<Vec<u8>>, Option<i64>, Option<i64>);

        let stored: Option<Row> = soma_staked_soma::table
            .select((
                soma_staked_soma::staked_soma_id,
                soma_staked_soma::cp_sequence_number,
                soma_staked_soma::owner,
                soma_staked_soma::pool_id,
                soma_staked_soma::stake_activation_epoch,
                soma_staked_soma::principal,
            ))
            .filter(soma_staked_soma::staked_soma_id.eq(&id_bytes))
            .order(soma_staked_soma::cp_sequence_number.desc())
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        // Filter out tombstones (owner IS NULL means deleted)
        match stored {
            Some(row) if row.2.is_some() => Ok(Some(StakedSoma {
                staked_soma_id: row.0,
                owner: row.2.unwrap(),
                pool_id: row.3.unwrap_or_default(),
                stake_activation_epoch: row.4.unwrap_or(0),
                principal: row.5.unwrap_or(0),
            })),
            _ => Ok(None),
        }
    }

    /// Query staked SOMA positions with pagination. Filter by owner and/or poolId.
    #[graphql(complexity = "5 + first.map(|f| f as usize).unwrap_or(20) * child_complexity")]
    async fn staked_somas(
        &self,
        ctx: &Context<'_>,
        owner: Option<String>,
        pool_id: Option<String>,
        first: Option<i32>,
        after: Option<String>,
    ) -> Result<Connection<String, StakedSoma>> {
        if owner.is_none() && pool_id.is_none() {
            return Err(Error::new("At least one of owner or poolId must be provided"));
        }

        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first.unwrap_or(config.default_page_size).min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        let owner_bytes: Option<Vec<u8>> = owner
            .as_deref()
            .map(|o| {
                let hex = o.strip_prefix("0x").unwrap_or(o);
                hex::decode(hex).map_err(|e| Error::new(format!("Invalid owner address: {e}")))
            })
            .transpose()?;
        let pool_bytes: Option<Vec<u8>> = pool_id
            .as_deref()
            .map(|p| {
                let hex = p.strip_prefix("0x").unwrap_or(p);
                hex::decode(hex).map_err(|e| Error::new(format!("Invalid pool ID: {e}")))
            })
            .transpose()?;

        use indexer_alt_schema::schema::soma_staked_soma;

        let after_id: Option<Vec<u8>> = after
            .as_deref()
            .map(|s| hex::decode(s).map_err(|e| Error::new(format!("Invalid cursor: {e}"))))
            .transpose()?;

        type Row = (Vec<u8>, i64, Option<Vec<u8>>, Option<Vec<u8>>, Option<i64>, Option<i64>);

        // Step 1: Find staked_soma_ids matching owner and/or pool_id.
        let mut id_query = soma_staked_soma::table
            .select(soma_staked_soma::staked_soma_id)
            .distinct()
            .into_boxed();

        if let Some(ref ob) = owner_bytes {
            id_query = id_query.filter(soma_staked_soma::owner.eq(ob));
        }
        if let Some(ref pb) = pool_bytes {
            id_query = id_query.filter(soma_staked_soma::pool_id.eq(pb));
        }

        if let Some(ref aid) = after_id {
            id_query = id_query.filter(soma_staked_soma::staked_soma_id.gt(aid));
        }

        let candidate_ids: Vec<Vec<u8>> =
            id_query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        if candidate_ids.is_empty() {
            return Ok(Connection::new(false, false));
        }

        // Step 2: Load ALL versions for those IDs (including tombstones) to
        // correctly determine which are still live.
        let rows: Vec<Row> = soma_staked_soma::table
            .select((
                soma_staked_soma::staked_soma_id,
                soma_staked_soma::cp_sequence_number,
                soma_staked_soma::owner,
                soma_staked_soma::pool_id,
                soma_staked_soma::stake_activation_epoch,
                soma_staked_soma::principal,
            ))
            .filter(soma_staked_soma::staked_soma_id.eq_any(&candidate_ids))
            .order((
                soma_staked_soma::staked_soma_id.asc(),
                soma_staked_soma::cp_sequence_number.desc(),
            ))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Deduplicate: keep only the latest cp per staked_soma_id, filter tombstones
        let mut seen = std::collections::HashSet::new();
        let mut deduped: Vec<StakedSoma> = Vec::new();
        for row in rows {
            if seen.contains(&row.0) {
                continue;
            }
            seen.insert(row.0.clone());
            // Filter out tombstones (owner IS NULL) and ownership changes
            // (latest row for this ID may show a different owner after transfer)
            if let Some(ref current_owner) = row.2 {
                if owner_bytes.as_ref().map_or(true, |ob| current_owner == ob) {
                    deduped.push(StakedSoma {
                        staked_soma_id: row.0,
                        owner: current_owner.clone(),
                        pool_id: row.3.unwrap_or_default(),
                        stake_activation_epoch: row.4.unwrap_or(0),
                        principal: row.5.unwrap_or(0),
                    });
                }
            }
        }

        // Paginate
        let has_next = deduped.len() as i64 > limit;
        let nodes: Vec<_> = deduped.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for s in nodes {
            let cursor = hex::encode(&s.staked_soma_id);
            connection.edges.push(Edge::new(cursor, s));
        }

        Ok(connection)
    }

    /// Get the total SOMA coin balance for an address.
    async fn balance(&self, ctx: &Context<'_>, address: String) -> Result<BigInt> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let addr_hex = address.strip_prefix("0x").unwrap_or(&address);
        let addr_bytes =
            hex::decode(addr_hex).map_err(|e| Error::new(format!("Invalid address: {e}")))?;

        use indexer_alt_schema::schema::{kv_objects, obj_info, obj_versions};

        // Step 1: Find Coin object IDs owned by this address (latest cp per object_id).
        // We load all obj_info rows for this owner where module='Coin', ordered by
        // object_id ASC, cp DESC, then deduplicate by object_id.
        type ObjInfoRow = (Vec<u8>, i64);

        let info_rows: Vec<ObjInfoRow> = obj_info::table
            .select((obj_info::object_id, obj_info::cp_sequence_number))
            .filter(obj_info::owner_id.eq(&addr_bytes))
            .filter(obj_info::module.eq("Coin"))
            .order((obj_info::object_id.asc(), obj_info::cp_sequence_number.desc()))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Deduplicate by object_id (keep latest cp)
        let mut seen_ids = std::collections::HashSet::new();
        let mut coin_object_ids: Vec<Vec<u8>> = Vec::new();
        for (oid, _cp) in &info_rows {
            if seen_ids.insert(oid.clone()) {
                coin_object_ids.push(oid.clone());
            }
        }

        if coin_object_ids.is_empty() {
            return Ok(BigInt(0));
        }

        // Step 1.5: Verify each candidate is still owned by this address.
        // The owner-filtered query above may return stale rows for coins that
        // were transferred to another address at a later checkpoint.
        let verify_rows: Vec<(Vec<u8>, Option<Vec<u8>>)> = obj_info::table
            .select((obj_info::object_id, obj_info::owner_id))
            .filter(obj_info::object_id.eq_any(&coin_object_ids))
            .order((obj_info::object_id.asc(), obj_info::cp_sequence_number.desc()))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let mut still_owned = std::collections::HashSet::new();
        let mut seen_verify = std::collections::HashSet::new();
        for (oid, owner) in &verify_rows {
            if seen_verify.insert(oid.clone()) {
                if owner.as_deref() == Some(addr_bytes.as_slice()) {
                    still_owned.insert(oid.clone());
                }
            }
        }
        coin_object_ids.retain(|oid| still_owned.contains(oid));

        if coin_object_ids.is_empty() {
            return Ok(BigInt(0));
        }

        // Step 2: For each coin object, get the latest version and extract balance.
        let kv = kv_loader(ctx);
        let mut total: i64 = 0;
        for oid in &coin_object_ids {
            let version: Option<i64> = obj_versions::table
                .select(obj_versions::object_version)
                .filter(obj_versions::object_id.eq(oid))
                .order(obj_versions::object_version.desc())
                .first(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?;

            let Some(ver) = version else { continue };

            // Step 3: Load object and extract coin balance.
            let balance: Option<u64> = if let Some(kv) = kv {
                let obj_id = types::object::ObjectID::from_bytes(oid.as_slice())
                    .map_err(|e| Error::new(e.to_string()))?;
                let obj: Option<types::object::Object> =
                    kv.get_object(&obj_id, ver as u64)
                        .await
                        .map_err(|e: anyhow::Error| Error::new(e.to_string()))?;
                obj.and_then(|o| o.as_coin())
            } else {
                let serialized: Option<Option<Vec<u8>>> = kv_objects::table
                    .select(kv_objects::serialized_object)
                    .filter(kv_objects::object_id.eq(oid))
                    .filter(kv_objects::object_version.eq(ver))
                    .first(conn.deref_mut())
                    .await
                    .optional()
                    .map_err(|e| Error::new(e.to_string()))?;
                serialized
                    .flatten()
                    .and_then(|bytes| bcs::from_bytes::<types::object::Object>(&bytes).ok())
                    .and_then(|obj| obj.as_coin())
            };

            if let Some(value) = balance {
                total = total.saturating_add(value as i64);
            }
        }

        Ok(BigInt(total))
    }

    /// Look up the epoch state for a given epoch (or latest).
    async fn epoch_state(
        &self,
        ctx: &Context<'_>,
        epoch: Option<i64>,
    ) -> Result<Option<EpochState>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_epoch_state;

        type Row = (i64, i64, i64, i64, i64, i32, i64, bool, i64, i64);

        let mut query = soma_epoch_state::table
            .select((
                soma_epoch_state::epoch,
                soma_epoch_state::emission_balance,
                soma_epoch_state::emission_per_epoch,
                soma_epoch_state::distribution_counter,
                soma_epoch_state::period_length,
                soma_epoch_state::decrease_rate,
                soma_epoch_state::protocol_fund_balance,
                soma_epoch_state::safe_mode,
                soma_epoch_state::safe_mode_accumulated_fees,
                soma_epoch_state::safe_mode_accumulated_emissions,
            ))
            .order(soma_epoch_state::epoch.desc())
            .into_boxed();

        if let Some(e) = epoch {
            query = query.filter(soma_epoch_state::epoch.eq(e));
        }

        let stored: Option<Row> = query
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(stored.map(|r| EpochState {
            epoch: r.0,
            emission_balance: r.1,
            emission_per_epoch: r.2,
            distribution_counter: r.3,
            period_length: r.4,
            decrease_rate: r.5,
            protocol_fund_balance: r.6,
            safe_mode: r.7,
            safe_mode_accumulated_fees: r.8,
            safe_mode_accumulated_emissions: r.9,
        }))
    }

    /// Query transactions with pagination and optional kind filter.
    #[graphql(complexity = "5 + first.map(|f| f as usize).unwrap_or(20) * child_complexity")]
    async fn transactions(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
        kind: Option<String>,
        exclude_kind: Option<String>,
        sender: Option<String>,
        epoch: Option<i64>,
    ) -> Result<Connection<String, TransactionDetail>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first.unwrap_or(config.default_page_size).min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_tx_details;

        let after_seq: Option<i64> = after
            .as_deref()
            .map(|s| s.parse().map_err(|e| Error::new(format!("Invalid cursor: {e}"))))
            .transpose()?;

        type Row = (i64, Vec<u8>, String, Vec<u8>, i64, i64, Option<String>);

        let mut query = soma_tx_details::table
            .select((
                soma_tx_details::tx_sequence_number,
                soma_tx_details::tx_digest,
                soma_tx_details::kind,
                soma_tx_details::sender,
                soma_tx_details::epoch,
                soma_tx_details::timestamp_ms,
                soma_tx_details::metadata_json,
            ))
            .order(soma_tx_details::tx_sequence_number.desc())
            .limit(limit + 1)
            .into_boxed();

        if let Some(ref kind_filter) = kind {
            query = query.filter(soma_tx_details::kind.eq(kind_filter));
        }
        if let Some(ref exclude) = exclude_kind {
            query = query.filter(soma_tx_details::kind.ne(exclude));
        }
        if let Some(ref sender_hex) = sender {
            let hex = sender_hex.strip_prefix("0x").unwrap_or(sender_hex);
            let bytes =
                hex::decode(hex).map_err(|e| Error::new(format!("Invalid sender address: {e}")))?;
            query = query.filter(soma_tx_details::sender.eq(bytes));
        }
        if let Some(e) = epoch {
            query = query.filter(soma_tx_details::epoch.eq(e));
        }
        if let Some(seq) = after_seq {
            query = query.filter(soma_tx_details::tx_sequence_number.lt(seq));
        }

        let results: Vec<Row> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        let has_next = results.len() as i64 > limit;
        let nodes: Vec<_> = results.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for row in nodes {
            let cursor = row.0.to_string();
            connection.edges.push(Edge::new(
                cursor,
                TransactionDetail {
                    tx_sequence_number: row.0,
                    tx_digest: row.1,
                    kind: row.2,
                    sender: row.3,
                    epoch: row.4,
                    timestamp_ms: row.5,
                    metadata_json: row.6,
                },
            ));
        }

        Ok(connection)
    }

    /// The range of checkpoints for which index-backed queries have complete data.
    async fn available_range(&self, ctx: &Context<'_>) -> Result<AvailableRange> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::cp_sequence_numbers;
        use indexer_pg_db::watermarks;

        // Read watermarks for a representative prunable pipeline (Tier B).
        let wm: Option<(i64, i64)> = watermarks::table
            .select((watermarks::reader_lo, watermarks::checkpoint_hi_inclusive))
            .filter(watermarks::pipeline.eq("kv_checkpoints"))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        match wm {
            Some((reader_lo, checkpoint_hi)) => {
                Ok(AvailableRange { first: reader_lo, last: checkpoint_hi })
            }
            None => {
                // No watermark = no pruning = everything available
                let max_cp: Option<i64> = cp_sequence_numbers::table
                    .select(cp_sequence_numbers::cp_sequence_number)
                    .order(cp_sequence_numbers::cp_sequence_number.desc())
                    .first(conn.deref_mut())
                    .await
                    .optional()
                    .map_err(|e| Error::new(e.to_string()))?;
                Ok(AvailableRange { first: 0, last: max_cp.unwrap_or(0) })
            }
        }
    }

    /// Query objects owned by an address.
    #[graphql(complexity = "5 + first.map(|f| f as usize).unwrap_or(20) * child_complexity")]
    async fn objects(
        &self,
        ctx: &Context<'_>,
        owner: String,
        first: Option<i32>,
        after: Option<String>,
    ) -> Result<Connection<String, GqlObject>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first.unwrap_or(config.default_page_size).min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        let owner_hex = owner.strip_prefix("0x").unwrap_or(&owner);
        let owner_bytes = hex::decode(owner_hex)
            .map_err(|e| Error::new(format!("Invalid owner address: {e}")))?;

        use indexer_alt_schema::schema::obj_info;

        // Cursor format: "hex_object_id"
        let after_bytes: Option<Vec<u8>> = after
            .as_deref()
            .map(|s| hex::decode(s).map_err(|e| Error::new(format!("Invalid cursor: {e}"))))
            .transpose()?;

        // Query obj_info for objects owned by this address.
        // Deduplicate by object_id (keep latest cp_sequence_number per object).
        type Row = (
            Vec<u8>,
            i64,
            Option<i16>,
            Option<Vec<u8>>,
            Option<Vec<u8>>,
            Option<String>,
            Option<String>,
        );

        let rows: Vec<Row> = obj_info::table
            .select((
                obj_info::object_id,
                obj_info::cp_sequence_number,
                obj_info::owner_kind,
                obj_info::owner_id,
                obj_info::package,
                obj_info::module,
                obj_info::name,
            ))
            .filter(obj_info::owner_id.eq(&owner_bytes))
            .filter(obj_info::owner_kind.eq(1_i16)) // Address-owned
            .order((obj_info::object_id.asc(), obj_info::cp_sequence_number.desc()))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Deduplicate: keep only the latest cp_sequence_number per object_id.
        // Filter out deleted objects (owner_id becomes None in latest row).
        let mut seen = std::collections::HashSet::new();
        let mut deduped: Vec<Row> = Vec::new();
        for row in rows {
            if seen.insert(row.0.clone()) {
                deduped.push(row);
            }
        }

        // Apply cursor pagination
        let start_idx = match &after_bytes {
            Some(id) => deduped.iter().position(|r| r.0 == *id).map(|i| i + 1).unwrap_or(0),
            None => 0,
        };

        let page: Vec<_> = deduped.into_iter().skip(start_idx).take(limit as usize + 1).collect();
        let has_next = page.len() as i64 > limit;
        let nodes: Vec<_> = page.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for row in nodes {
            let cursor = hex::encode(&row.0);
            let owner_kind_str = row.2.map(|k| match k {
                0 => "Immutable",
                1 => "Address",
                2 => "Object",
                3 => "Shared",
                _ => "Unknown",
            });
            let object_type = row.5.clone(); // module as type
            connection.edges.push(Edge::new(
                cursor,
                GqlObject {
                    object_id: row.0,
                    object_version: 0, // Not tracked in obj_info
                    serialized_object_bcs: None,
                    owner_kind: owner_kind_str.map(String::from),
                    owner_id: row.3,
                    object_type,
                },
            ));
        }

        Ok(connection)
    }

    /// Query validators at a given epoch (or latest).
    #[graphql(complexity = "5 + first.map(|f| f as usize).unwrap_or(20) * child_complexity")]
    async fn validators(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
        epoch: Option<i64>,
    ) -> Result<Connection<String, Validator>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first.unwrap_or(config.default_page_size).min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_validators;

        // Epoch discovery: use provided epoch or latest
        let epoch_num = match epoch {
            Some(e) => e,
            None => soma_validators::table
                .select(soma_validators::epoch)
                .order(soma_validators::epoch.desc())
                .first::<i64>(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?
                .ok_or_else(|| Error::new("No validators found"))?,
        };

        // Cursor: hex-encoded address
        let after_bytes: Option<Vec<u8>> = after
            .as_deref()
            .map(|s| hex::decode(s).map_err(|e| Error::new(format!("Invalid cursor: {e}"))))
            .transpose()?;

        type Row = (
            Vec<u8>,
            i64,
            i64,
            i64,
            i64,
            Vec<u8>,
            i64,
            i64,
            Option<String>,
            Option<String>,
            Option<String>,
        );

        let mut query = soma_validators::table
            .select((
                soma_validators::address,
                soma_validators::epoch,
                soma_validators::voting_power,
                soma_validators::commission_rate,
                soma_validators::next_epoch_commission_rate,
                soma_validators::staking_pool_id,
                soma_validators::stake,
                soma_validators::pending_stake,
                soma_validators::name,
                soma_validators::network_address,
                soma_validators::proxy_address,
            ))
            .filter(soma_validators::epoch.eq(epoch_num))
            .order(soma_validators::stake.desc())
            .limit(limit + 1)
            .into_boxed();

        if let Some(ref addr) = after_bytes {
            // For cursor pagination with ordering by stake, we need position-based cursoring.
            // We'll use address as cursor and skip past it.
            query = query.filter(soma_validators::address.gt(addr));
        }

        let results: Vec<Row> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        let has_next = results.len() as i64 > limit;
        let nodes: Vec<_> = results.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for row in nodes {
            let cursor = hex::encode(&row.0);
            connection.edges.push(Edge::new(
                cursor,
                Validator {
                    address: row.0,
                    epoch: row.1,
                    voting_power: row.2,
                    commission_rate: row.3,
                    next_epoch_commission_rate: row.4,
                    staking_pool_id: row.5,
                    stake: row.6,
                    pending_stake: row.7,
                    name: row.8,
                    network_address: row.9,
                    proxy_address: row.10,
                },
            ));
        }

        Ok(connection)
    }

    /// Look up a single validator by address at a given epoch (or latest).
    async fn validator(
        &self,
        ctx: &Context<'_>,
        address: String,
        epoch: Option<i64>,
    ) -> Result<Option<Validator>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let addr_hex = address.strip_prefix("0x").unwrap_or(&address);
        let addr_bytes =
            hex::decode(addr_hex).map_err(|e| Error::new(format!("Invalid address: {e}")))?;

        use indexer_alt_schema::schema::soma_validators;

        type Row = (
            Vec<u8>,
            i64,
            i64,
            i64,
            i64,
            Vec<u8>,
            i64,
            i64,
            Option<String>,
            Option<String>,
            Option<String>,
        );

        let mut query = soma_validators::table
            .select((
                soma_validators::address,
                soma_validators::epoch,
                soma_validators::voting_power,
                soma_validators::commission_rate,
                soma_validators::next_epoch_commission_rate,
                soma_validators::staking_pool_id,
                soma_validators::stake,
                soma_validators::pending_stake,
                soma_validators::name,
                soma_validators::network_address,
                soma_validators::proxy_address,
            ))
            .filter(soma_validators::address.eq(&addr_bytes))
            .order(soma_validators::epoch.desc())
            .limit(1)
            .into_boxed();

        if let Some(e) = epoch {
            query = query.filter(soma_validators::epoch.eq(e));
        }

        let result: Option<Row> = query
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(result.map(|row| Validator {
            address: row.0,
            epoch: row.1,
            voting_power: row.2,
            commission_rate: row.3,
            next_epoch_commission_rate: row.4,
            staking_pool_id: row.5,
            stake: row.6,
            pending_stake: row.7,
            name: row.8,
            network_address: row.9,
            proxy_address: row.10,
        }))
    }

    /// Network-wide metrics (TPS, totals).
    async fn network_metrics(&self, ctx: &Context<'_>) -> Result<NetworkMetrics> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::{cp_sequence_numbers, kv_checkpoints, soma_validators};

        // Total transactions: txLo of latest checkpoint
        let latest_cp: Option<(i64, i64)> = cp_sequence_numbers::table
            .select((cp_sequence_numbers::cp_sequence_number, cp_sequence_numbers::tx_lo))
            .order(cp_sequence_numbers::cp_sequence_number.desc())
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        let (total_checkpoints, total_transactions) = match latest_cp {
            Some((cp, tx_lo)) => (cp + 1, tx_lo),
            None => (0, 0),
        };

        // Total validators at latest epoch
        let latest_validator_epoch: Option<i64> = soma_validators::table
            .select(soma_validators::epoch)
            .order(soma_validators::epoch.desc())
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        let total_validators = match latest_validator_epoch {
            Some(e) => soma_validators::table
                .filter(soma_validators::epoch.eq(e))
                .count()
                .get_result::<i64>(conn.deref_mut())
                .await
                .map_err(|e| Error::new(e.to_string()))?,
            None => 0,
        };

        // TPS: computed from last 10 checkpoints' timestamps.
        // Checkpoint timestamps may be null on testnet, so this is best-effort.
        let recent_cps: Vec<(i64, i64)> = cp_sequence_numbers::table
            .select((cp_sequence_numbers::cp_sequence_number, cp_sequence_numbers::tx_lo))
            .order(cp_sequence_numbers::cp_sequence_number.desc())
            .limit(11)
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Try to get timestamps from kv_checkpoints to compute TPS
        let tps: Option<f64> = if recent_cps.len() >= 2 {
            // Get checkpoint summaries and try to extract timestamps
            let cp_seqs: Vec<i64> = recent_cps.iter().map(|(cp, _)| *cp).collect();
            let summaries: Vec<(i64, Vec<u8>)> = kv_checkpoints::table
                .select((kv_checkpoints::sequence_number, kv_checkpoints::checkpoint_summary))
                .filter(kv_checkpoints::sequence_number.eq_any(&cp_seqs))
                .order(kv_checkpoints::sequence_number.desc())
                .load(conn.deref_mut())
                .await
                .map_err(|e| Error::new(e.to_string()))?;

            // Try to deserialize and get timestamps
            let mut timed_cps: Vec<(i64, u64)> = Vec::new();
            for (seq, bcs_data) in &summaries {
                if let Ok(summary) =
                    bcs::from_bytes::<types::checkpoints::CheckpointSummary>(bcs_data)
                {
                    if summary.timestamp_ms > 0 {
                        // Find tx_lo for this checkpoint
                        if let Some((_, tx_lo)) = recent_cps.iter().find(|(cp, _)| cp == seq) {
                            timed_cps.push((*tx_lo, summary.timestamp_ms));
                        }
                    }
                }
            }

            if timed_cps.len() >= 2 {
                let newest = &timed_cps[0];
                let oldest = &timed_cps[timed_cps.len() - 1];
                let tx_diff = (newest.0 as f64) - (oldest.0 as f64);
                let time_diff_secs = (newest.1 as f64 - oldest.1 as f64) / 1000.0;
                if time_diff_secs > 0.0 { Some(tx_diff / time_diff_secs) } else { None }
            } else {
                None
            }
        } else {
            None
        };

        Ok(NetworkMetrics { tps, total_transactions, total_checkpoints, total_validators })
    }
}
