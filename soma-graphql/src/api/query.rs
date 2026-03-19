// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::ops::DerefMut;
use std::sync::Arc;

use async_graphql::connection::{Connection, Edge};
use async_graphql::{Context, Error, Object, Result};
use diesel::ExpressionMethods;
use diesel::NullableExpressionMethods;
use diesel::OptionalExtension;
use diesel::QueryDsl;
use diesel::dsl::count_star;
use diesel_async::RunQueryDsl;

use crate::api::scalars::BigInt;
use crate::api::types::address::Address;
use crate::api::types::aggregates::{
    ModelAggregates, ModelStats, RewardAggregates, SubmitterStats, TargetAggregates,
    TargetScoreAggregates,
};
use crate::api::types::available_range::AvailableRange;
use crate::api::types::checkpoint::Checkpoint;
use crate::api::types::epoch::Epoch;
use crate::api::types::epoch_state::EpochState;
use crate::api::types::model::{Model, ModelFilter};
use crate::api::types::network_metrics::NetworkMetrics;
use crate::api::types::object::Object as GqlObject;
use crate::api::types::reward::{Reward, RewardBalance};
use crate::api::types::service_config::ServiceConfig;
use crate::api::types::staked_soma::StakedSoma;
use crate::api::types::target::{Target, TargetFilter};
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

    /// Look up a target by ID. Returns the latest version.
    async fn target(&self, ctx: &Context<'_>, target_id: String) -> Result<Option<Target>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let id_hex = target_id.strip_prefix("0x").unwrap_or(&target_id);
        let id_bytes =
            hex::decode(id_hex).map_err(|e| Error::new(format!("Invalid target ID: {e}")))?;

        use indexer_alt_schema::schema::soma_targets;

        // Split into nested tuples to stay within Diesel's 16-element tuple limit.
        type RowA = (Vec<u8>, i64, i64, String, Option<Vec<u8>>, Option<Vec<u8>>, i64, i64, i32);
        type RowB = (
            Option<f64>,
            Option<f64>,
            Option<Vec<u8>>,
            Option<i64>,
            f64,
            String,
            Option<String>,
            Option<Vec<u8>>,
            Option<i64>,
        );

        let stored: Option<(RowA, RowB)> = soma_targets::table
            .select((
                (
                    soma_targets::target_id,
                    soma_targets::cp_sequence_number,
                    soma_targets::epoch,
                    soma_targets::status,
                    soma_targets::submitter,
                    soma_targets::winning_model_id,
                    soma_targets::reward_pool,
                    soma_targets::bond_amount,
                    soma_targets::report_count,
                ),
                (
                    soma_targets::winning_distance_score,
                    soma_targets::winning_loss_score,
                    soma_targets::winning_model_owner,
                    soma_targets::fill_epoch,
                    soma_targets::distance_threshold,
                    soma_targets::model_ids_json,
                    soma_targets::winning_data_url,
                    soma_targets::winning_data_checksum,
                    soma_targets::winning_data_size,
                ),
            ))
            .filter(soma_targets::target_id.eq(&id_bytes))
            .order(soma_targets::cp_sequence_number.desc())
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(stored.map(|(a, b)| target_from_row(a, b)))
    }

    /// Query targets with pagination and optional filters.
    #[graphql(complexity = "5 + first.map(|f| f as usize).unwrap_or(20) * child_complexity")]
    async fn targets(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
        filter: Option<TargetFilter>,
    ) -> Result<Connection<String, Target>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first.unwrap_or(config.default_page_size).min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_targets;

        let after_cp: Option<i64> = match &after {
            Some(cursor) => {
                let parts: Vec<&str> = cursor.splitn(2, ':').collect();
                if parts.len() != 2 {
                    return Err(Error::new("Invalid cursor format"));
                }
                let cp: i64 =
                    parts[1].parse().map_err(|e| Error::new(format!("Invalid cursor cp: {e}")))?;
                Some(cp)
            }
            None => None,
        };

        // Split into nested tuples to stay within Diesel's 16-element tuple limit.
        type RowA = (Vec<u8>, i64, i64, String, Option<Vec<u8>>, Option<Vec<u8>>, i64, i64, i32);
        type RowB = (
            Option<f64>,
            Option<f64>,
            Option<Vec<u8>>,
            Option<i64>,
            f64,
            String,
            Option<String>,
            Option<Vec<u8>>,
            Option<i64>,
        );

        let mut query = soma_targets::table
            .select((
                (
                    soma_targets::target_id,
                    soma_targets::cp_sequence_number,
                    soma_targets::epoch,
                    soma_targets::status,
                    soma_targets::submitter,
                    soma_targets::winning_model_id,
                    soma_targets::reward_pool,
                    soma_targets::bond_amount,
                    soma_targets::report_count,
                ),
                (
                    soma_targets::winning_distance_score,
                    soma_targets::winning_loss_score,
                    soma_targets::winning_model_owner,
                    soma_targets::fill_epoch,
                    soma_targets::distance_threshold,
                    soma_targets::model_ids_json,
                    soma_targets::winning_data_url,
                    soma_targets::winning_data_checksum,
                    soma_targets::winning_data_size,
                ),
            ))
            .order(soma_targets::cp_sequence_number.desc())
            .limit(limit + 1)
            .into_boxed();

        if let Some(ref filter) = filter {
            if let Some(ref status) = filter.status {
                query = query.filter(soma_targets::status.eq(status));
            }
            if let Some(epoch) = filter.epoch {
                query = query.filter(soma_targets::epoch.eq(epoch));
            }
            if let Some(ref submitter_hex) = filter.submitter {
                let hex = submitter_hex.strip_prefix("0x").unwrap_or(submitter_hex);
                let bytes = hex::decode(hex)
                    .map_err(|e| Error::new(format!("Invalid submitter address: {e}")))?;
                query = query.filter(soma_targets::submitter.eq(bytes));
            }
            if let Some(ref model_hex) = filter.winning_model_id {
                let hex = model_hex.strip_prefix("0x").unwrap_or(model_hex);
                let bytes = hex::decode(hex)
                    .map_err(|e| Error::new(format!("Invalid winning_model_id: {e}")))?;
                query = query.filter(soma_targets::winning_model_id.eq(bytes));
            }
            if let Some(fe) = filter.fill_epoch {
                query = query.filter(soma_targets::fill_epoch.eq(fe));
            }
            if let Some(ref owner_hex) = filter.winning_model_owner {
                let hex = owner_hex.strip_prefix("0x").unwrap_or(owner_hex);
                let bytes = hex::decode(hex)
                    .map_err(|e| Error::new(format!("Invalid winning_model_owner: {e}")))?;
                query = query.filter(soma_targets::winning_model_owner.eq(bytes));
            }
        }

        if let Some(acp) = after_cp {
            query = query.filter(soma_targets::cp_sequence_number.lt(acp));
        }

        let results: Vec<(RowA, RowB)> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        let has_next = results.len() as i64 > limit;
        let nodes: Vec<_> = results.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for (a, b) in nodes {
            let cursor = format!("{}:{}", hex::encode(&a.0), a.1);
            connection.edges.push(Edge::new(cursor, target_from_row(a, b)));
        }

        Ok(connection)
    }

    /// Look up a model by ID. Returns the snapshot at the given epoch, or latest.
    async fn model(
        &self,
        ctx: &Context<'_>,
        model_id: String,
        epoch: Option<i64>,
    ) -> Result<Option<Model>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let id_hex = model_id.strip_prefix("0x").unwrap_or(&model_id);
        let id_bytes =
            hex::decode(id_hex).map_err(|e| Error::new(format!("Invalid model ID: {e}")))?;

        use indexer_alt_schema::schema::soma_models;

        let mut query = soma_models::table
            .select(model_select())
            .filter(soma_models::model_id.eq(&id_bytes))
            .order(soma_models::epoch.desc())
            .into_boxed();

        if let Some(e) = epoch {
            query = query.filter(soma_models::epoch.eq(e));
        }

        let stored: Option<(ModelRowA, ModelRowB, ModelRowC)> = query
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(stored.map(|(a, b, c)| model_from_row(a, b, c)))
    }

    /// Query models with pagination and optional filters.
    #[graphql(complexity = "5 + first.map(|f| f as usize).unwrap_or(20) * child_complexity")]
    async fn models(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
        epoch: Option<i64>,
        filter: Option<ModelFilter>,
    ) -> Result<Connection<String, Model>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first.unwrap_or(config.default_page_size).min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_models;

        let epoch_num = match epoch {
            Some(e) => e,
            None => soma_models::table
                .select(soma_models::epoch)
                .order(soma_models::epoch.desc())
                .first::<i64>(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?
                .unwrap_or(0),
        };

        let after_id: Option<Vec<u8>> = after
            .as_deref()
            .map(|s| hex::decode(s).map_err(|e| Error::new(format!("Invalid cursor: {e}"))))
            .transpose()?;

        let mut query = soma_models::table
            .select(model_select())
            .filter(soma_models::epoch.eq(epoch_num))
            .order(soma_models::model_id.asc())
            .limit(limit + 1)
            .into_boxed();

        if let Some(ref aid) = after_id {
            query = query.filter(soma_models::model_id.gt(aid));
        }

        if let Some(ref filter) = filter {
            if let Some(ref status) = filter.status {
                query = query.filter(soma_models::status.eq(status));
            }
            if let Some(ref owner_hex) = filter.owner {
                let hex = owner_hex.strip_prefix("0x").unwrap_or(owner_hex);
                let bytes = hex::decode(hex)
                    .map_err(|e| Error::new(format!("Invalid owner address: {e}")))?;
                query = query.filter(soma_models::owner.eq(bytes));
            }
            if let Some(min) = filter.min_stake {
                query = query.filter(soma_models::stake.ge(min));
            }
            if let Some(max) = filter.max_stake {
                query = query.filter(soma_models::stake.le(max));
            }
        }

        let results: Vec<(ModelRowA, ModelRowB, ModelRowC)> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        let has_next = results.len() as i64 > limit;
        let nodes: Vec<_> = results.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for (a, b, c) in nodes {
            let cursor = hex::encode(&a.0);
            connection.edges.push(Edge::new(cursor, model_from_row(a, b, c)));
        }

        Ok(connection)
    }

    /// Query rewards for a specific epoch and optionally a specific target or recipient.
    async fn rewards(
        &self,
        ctx: &Context<'_>,
        epoch: i64,
        target_id: Option<String>,
        recipient: Option<String>,
    ) -> Result<Vec<Reward>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::{soma_reward_balances, soma_rewards};

        // If filtering by recipient, find matching target_ids first
        let recipient_target_ids: Option<Vec<Vec<u8>>> = match &recipient {
            Some(hex_str) => {
                let hex = hex_str.strip_prefix("0x").unwrap_or(hex_str);
                let bytes = hex::decode(hex)
                    .map_err(|e| Error::new(format!("Invalid recipient address: {e}")))?;
                let ids: Vec<Vec<u8>> = soma_reward_balances::table
                    .select(soma_reward_balances::target_id)
                    .filter(soma_reward_balances::epoch.eq(epoch))
                    .filter(soma_reward_balances::recipient.eq(&bytes))
                    .load(conn.deref_mut())
                    .await
                    .map_err(|e| Error::new(e.to_string()))?;
                Some(ids)
            }
            None => None,
        };

        type RewardRow = (Vec<u8>, i64, i64, Vec<u8>);

        let mut query = soma_rewards::table
            .select((
                soma_rewards::target_id,
                soma_rewards::cp_sequence_number,
                soma_rewards::epoch,
                soma_rewards::tx_digest,
            ))
            .filter(soma_rewards::epoch.eq(epoch))
            .order(soma_rewards::cp_sequence_number.asc())
            .limit(100)
            .into_boxed();

        if let Some(ref tid) = target_id {
            let id_hex = tid.strip_prefix("0x").unwrap_or(tid);
            let id_bytes =
                hex::decode(id_hex).map_err(|e| Error::new(format!("Invalid target ID: {e}")))?;
            query = query.filter(soma_rewards::target_id.eq(id_bytes));
        }

        if let Some(ref ids) = recipient_target_ids {
            query = query.filter(soma_rewards::target_id.eq_any(ids));
        }

        let reward_rows: Vec<RewardRow> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        if reward_rows.is_empty() {
            return Ok(vec![]);
        }

        // Load all balance rows for matching targets in this epoch
        let target_ids: Vec<Vec<u8>> = reward_rows.iter().map(|r| r.0.clone()).collect();

        type BalanceRow = (Vec<u8>, Vec<u8>, i64);

        let balance_rows: Vec<BalanceRow> = soma_reward_balances::table
            .select((
                soma_reward_balances::target_id,
                soma_reward_balances::recipient,
                soma_reward_balances::amount,
            ))
            .filter(soma_reward_balances::target_id.eq_any(&target_ids))
            .filter(soma_reward_balances::epoch.eq(epoch))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Group balances by target_id
        let mut balance_map: std::collections::HashMap<Vec<u8>, Vec<RewardBalance>> =
            std::collections::HashMap::new();
        for (tid, recipient, amount) in balance_rows {
            balance_map.entry(tid).or_default().push(RewardBalance { recipient, amount });
        }

        Ok(reward_rows
            .into_iter()
            .map(|r| Reward {
                target_id: r.0.clone(),
                cp_sequence_number: r.1,
                epoch: r.2,
                tx_digest: r.3,
                balances: balance_map.remove(&r.0).unwrap_or_default(),
            })
            .collect())
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

    /// Aggregate statistics for models at a given epoch (or latest).
    async fn model_aggregates(
        &self,
        ctx: &Context<'_>,
        epoch: Option<i64>,
    ) -> Result<ModelAggregates> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_models;

        let epoch_num = match epoch {
            Some(e) => e,
            None => soma_models::table
                .select(soma_models::epoch)
                .order(soma_models::epoch.desc())
                .first::<i64>(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?
                .unwrap_or(0),
        };

        // Total count
        let total_count: i64 = soma_models::table
            .filter(soma_models::epoch.eq(epoch_num))
            .count()
            .get_result(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Active count
        let active_count: i64 = soma_models::table
            .filter(soma_models::epoch.eq(epoch_num))
            .filter(soma_models::status.eq("active"))
            .count()
            .get_result(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Total stake
        let total_stake: Option<i64> = soma_models::table
            .filter(soma_models::epoch.eq(epoch_num))
            .select(diesel::dsl::sql::<diesel::sql_types::Nullable<diesel::sql_types::BigInt>>(
                "COALESCE(SUM(stake), 0)::BIGINT",
            ))
            .first(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let total_stake = total_stake.unwrap_or(0);
        let avg_stake = if total_count > 0 { total_stake as f64 / total_count as f64 } else { 0.0 };

        Ok(ModelAggregates { total_count, total_stake, avg_stake, active_count })
    }

    /// Aggregate statistics for targets at a given epoch (or latest).
    async fn target_aggregates(
        &self,
        ctx: &Context<'_>,
        epoch: Option<i64>,
    ) -> Result<TargetAggregates> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_targets;

        let epoch_num = match epoch {
            Some(e) => e,
            None => soma_targets::table
                .select(soma_targets::epoch)
                .order(soma_targets::epoch.desc())
                .first::<i64>(conn.deref_mut())
                .await
                .optional()
                .map_err(|e| Error::new(e.to_string()))?
                .unwrap_or(0),
        };

        // Get latest version of each target in this epoch by loading all rows
        // and deduplicating by target_id (keeping highest cp).
        type AggRow = (Vec<u8>, i64, String, i64);

        let rows: Vec<AggRow> = soma_targets::table
            .select((
                soma_targets::target_id,
                soma_targets::cp_sequence_number,
                soma_targets::status,
                soma_targets::reward_pool,
            ))
            .filter(soma_targets::epoch.eq(epoch_num))
            .order((soma_targets::target_id.asc(), soma_targets::cp_sequence_number.desc()))
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Deduplicate by target_id, keeping latest cp
        let mut seen = std::collections::HashSet::new();
        let mut total_count: i64 = 0;
        let mut open_count: i64 = 0;
        let mut filled_count: i64 = 0;
        let mut claimed_count: i64 = 0;
        let mut total_reward_pool: i64 = 0;

        for (tid, _cp, status, reward_pool) in &rows {
            if !seen.insert(tid.clone()) {
                continue;
            }
            total_count += 1;
            total_reward_pool = total_reward_pool.saturating_add(*reward_pool);
            match status.as_str() {
                "open" => open_count += 1,
                "filled" => filled_count += 1,
                "claimed" => claimed_count += 1,
                _ => {}
            }
        }

        Ok(TargetAggregates {
            total_count,
            open_count,
            filled_count,
            claimed_count,
            total_reward_pool,
        })
    }

    /// Aggregate statistics for rewards at a given epoch.
    async fn reward_aggregates(&self, ctx: &Context<'_>, epoch: i64) -> Result<RewardAggregates> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::{soma_reward_balances, soma_rewards};

        // Count reward claims
        let total_count: i64 = soma_rewards::table
            .filter(soma_rewards::epoch.eq(epoch))
            .count()
            .get_result(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        // Sum amounts from reward balances
        let total_amount: Option<i64> = soma_reward_balances::table
            .filter(soma_reward_balances::epoch.eq(epoch))
            .select(diesel::dsl::sql::<diesel::sql_types::Nullable<diesel::sql_types::BigInt>>(
                "COALESCE(SUM(amount), 0)::BIGINT",
            ))
            .first(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(RewardAggregates { total_count, total_amount: total_amount.unwrap_or(0) })
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

        type Row = (i64, i64, i64, f64, i64, i64, i64, i64, bool, i64, i64);

        let mut query = soma_epoch_state::table
            .select((
                soma_epoch_state::epoch,
                soma_epoch_state::emission_balance,
                soma_epoch_state::emission_per_epoch,
                soma_epoch_state::distance_threshold,
                soma_epoch_state::targets_generated_this_epoch,
                soma_epoch_state::hits_this_epoch,
                soma_epoch_state::hits_ema,
                soma_epoch_state::reward_per_target,
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
            distance_threshold: r.3,
            targets_generated_this_epoch: r.4,
            hits_this_epoch: r.5,
            hits_ema: r.6,
            reward_per_target: r.7,
            safe_mode: r.8,
            safe_mode_accumulated_fees: r.9,
            safe_mode_accumulated_emissions: r.10,
        }))
    }

    /// Query the history of a model across epochs.
    async fn model_history(
        &self,
        ctx: &Context<'_>,
        model_id: String,
        from_epoch: Option<i64>,
        to_epoch: Option<i64>,
    ) -> Result<Vec<Model>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let id_hex = model_id.strip_prefix("0x").unwrap_or(&model_id);
        let id_bytes =
            hex::decode(id_hex).map_err(|e| Error::new(format!("Invalid model ID: {e}")))?;

        use indexer_alt_schema::schema::soma_models;

        let mut query = soma_models::table
            .select(model_select())
            .filter(soma_models::model_id.eq(&id_bytes))
            .order(soma_models::epoch.asc())
            .into_boxed();

        if let Some(from) = from_epoch {
            query = query.filter(soma_models::epoch.ge(from));
        }
        if let Some(to) = to_epoch {
            query = query.filter(soma_models::epoch.le(to));
        }

        let results: Vec<(ModelRowA, ModelRowB, ModelRowC)> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        Ok(results.into_iter().map(|(a, b, c)| model_from_row(a, b, c)).collect())
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

    /// Aggregate score statistics for filled targets at a given epoch (or all epochs).
    async fn target_score_aggregates(
        &self,
        ctx: &Context<'_>,
        epoch: Option<i64>,
    ) -> Result<TargetScoreAggregates> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_targets;

        // Load filled targets, deduplicate by target_id (keep latest cp).
        type Row = (Vec<u8>, i64, Option<f64>, Option<f64>, i64, Option<i64>);

        let mut query = soma_targets::table
            .select((
                soma_targets::target_id,
                soma_targets::cp_sequence_number,
                soma_targets::winning_loss_score,
                soma_targets::winning_distance_score,
                soma_targets::reward_pool,
                soma_targets::winning_data_size,
            ))
            .filter(soma_targets::status.eq("filled"))
            .order((soma_targets::target_id.asc(), soma_targets::cp_sequence_number.desc()))
            .into_boxed();

        if let Some(e) = epoch {
            query = query.filter(soma_targets::epoch.eq(e));
        }

        let rows: Vec<Row> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        let mut seen = std::collections::HashSet::new();
        let mut loss_sum = 0.0_f64;
        let mut loss_count = 0_i64;
        let mut dist_sum = 0.0_f64;
        let mut dist_count = 0_i64;
        let mut total_data_size = 0_i64;
        let mut filled_count = 0_i64;

        for (tid, _cp, loss, dist, _rp, data_size) in &rows {
            if !seen.insert(tid.clone()) {
                continue;
            }
            filled_count += 1;
            if let Some(l) = loss {
                loss_sum += l;
                loss_count += 1;
            }
            if let Some(d) = dist {
                dist_sum += d;
                dist_count += 1;
            }
            if let Some(ds) = data_size {
                total_data_size = total_data_size.saturating_add(*ds);
            }
        }

        Ok(TargetScoreAggregates {
            avg_loss_score: if loss_count > 0 { Some(loss_sum / loss_count as f64) } else { None },
            avg_distance_score: if dist_count > 0 {
                Some(dist_sum / dist_count as f64)
            } else {
                None
            },
            total_data_size,
            filled_count,
        })
    }

    /// Leaderboard of data submitters, ranked by number of filled targets.
    #[graphql(complexity = "10 + first.map(|f| f as usize).unwrap_or(20) * child_complexity")]
    async fn submitter_leaderboard(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
        epoch: Option<i64>,
    ) -> Result<Connection<String, SubmitterStats>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first.unwrap_or(config.default_page_size).min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_targets;

        // Load all filled targets, deduplicate by target_id, then aggregate by submitter.
        type Row = (Vec<u8>, i64, Option<Vec<u8>>, Option<f64>, Option<f64>, i64, Option<i64>);

        let mut query = soma_targets::table
            .select((
                soma_targets::target_id,
                soma_targets::cp_sequence_number,
                soma_targets::submitter,
                soma_targets::winning_distance_score,
                soma_targets::winning_loss_score,
                soma_targets::reward_pool,
                soma_targets::winning_data_size,
            ))
            .filter(soma_targets::status.eq("filled"))
            .order((soma_targets::target_id.asc(), soma_targets::cp_sequence_number.desc()))
            .into_boxed();

        if let Some(e) = epoch {
            query = query.filter(soma_targets::epoch.eq(e));
        }

        let rows: Vec<Row> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        // Deduplicate by target_id, aggregate by submitter
        let mut seen = std::collections::HashSet::new();
        let mut submitter_map: std::collections::HashMap<
            Vec<u8>,
            (i64, f64, i64, f64, i64, i64, i64),
        > = std::collections::HashMap::new();
        // Value: (target_count, dist_sum, dist_count, loss_sum, loss_count, total_reward, total_data_size)

        for (tid, _cp, submitter, dist, loss, rp, data_size) in &rows {
            if !seen.insert(tid.clone()) {
                continue;
            }
            let Some(sub) = submitter else { continue };
            let entry = submitter_map.entry(sub.clone()).or_default();
            entry.0 += 1;
            if let Some(d) = dist {
                entry.1 += d;
                entry.2 += 1;
            }
            if let Some(l) = loss {
                entry.3 += l;
                entry.4 += 1;
            }
            entry.5 = entry.5.saturating_add(*rp);
            if let Some(ds) = data_size {
                entry.6 = entry.6.saturating_add(*ds);
            }
        }

        // Sort by target_count desc
        let mut entries: Vec<_> = submitter_map.into_iter().collect();
        entries.sort_by(|a, b| b.1.0.cmp(&a.1.0));

        // Apply cursor pagination
        let after_addr: Option<Vec<u8>> = after
            .as_deref()
            .map(|s| hex::decode(s).map_err(|e| Error::new(format!("Invalid cursor: {e}"))))
            .transpose()?;

        let start_idx = match &after_addr {
            Some(addr) => entries.iter().position(|(k, _)| k == addr).map(|i| i + 1).unwrap_or(0),
            None => 0,
        };

        let page: Vec<_> = entries.into_iter().skip(start_idx).take(limit as usize + 1).collect();
        let has_next = page.len() as i64 > limit;
        let nodes: Vec<_> = page.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for (sub, (tc, ds, dc, ls, lc, tr, tds)) in nodes {
            let cursor = hex::encode(&sub);
            connection.edges.push(Edge::new(
                cursor,
                SubmitterStats {
                    submitter: sub,
                    target_count: tc,
                    avg_distance_score: if dc > 0 { Some(ds / dc as f64) } else { None },
                    avg_loss_score: if lc > 0 { Some(ls / lc as f64) } else { None },
                    total_reward: tr,
                    total_data_size: tds,
                },
            ));
        }

        Ok(connection)
    }

    /// Stats for a single data submitter.
    async fn submitter_stats(
        &self,
        ctx: &Context<'_>,
        address: String,
        epoch: Option<i64>,
    ) -> Result<Option<SubmitterStats>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let addr_bytes =
            hex::decode(&address).map_err(|e| Error::new(format!("Invalid address: {e}")))?;

        use indexer_alt_schema::schema::soma_targets;

        type Row = (Vec<u8>, i64, Option<f64>, Option<f64>, i64, Option<i64>);

        let mut query = soma_targets::table
            .select((
                soma_targets::target_id,
                soma_targets::cp_sequence_number,
                soma_targets::winning_distance_score,
                soma_targets::winning_loss_score,
                soma_targets::reward_pool,
                soma_targets::winning_data_size,
            ))
            .filter(soma_targets::status.eq("filled"))
            .filter(soma_targets::submitter.eq(&addr_bytes))
            .order((
                soma_targets::target_id.asc(),
                soma_targets::cp_sequence_number.desc(),
            ))
            .into_boxed();

        if let Some(e) = epoch {
            query = query.filter(soma_targets::epoch.eq(e));
        }

        let rows: Vec<Row> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        if rows.is_empty() {
            return Ok(None);
        }

        // Deduplicate by target_id, aggregate
        let mut seen = std::collections::HashSet::new();
        let (mut target_count, mut dist_sum, mut dist_count, mut loss_sum, mut loss_count) =
            (0i64, 0.0f64, 0i64, 0.0f64, 0i64);
        let (mut total_reward, mut total_data_size) = (0i64, 0i64);

        for (tid, _cp, dist, loss, rp, data_size) in &rows {
            if !seen.insert(tid.clone()) {
                continue;
            }
            target_count += 1;
            if let Some(d) = dist {
                dist_sum += d;
                dist_count += 1;
            }
            if let Some(l) = loss {
                loss_sum += l;
                loss_count += 1;
            }
            total_reward = total_reward.saturating_add(*rp);
            if let Some(ds) = data_size {
                total_data_size = total_data_size.saturating_add(*ds);
            }
        }

        Ok(Some(SubmitterStats {
            submitter: addr_bytes,
            target_count,
            avg_distance_score: if dist_count > 0 {
                Some(dist_sum / dist_count as f64)
            } else {
                None
            },
            avg_loss_score: if loss_count > 0 {
                Some(loss_sum / loss_count as f64)
            } else {
                None
            },
            total_reward,
            total_data_size,
        }))
    }

    /// Leaderboard of models, ranked by number of targets won.
    /// Includes win rate (targets won / targets assigned), average scores, and total rewards.
    #[graphql(complexity = "10 + first.map(|f| f as usize).unwrap_or(20) * child_complexity")]
    async fn model_leaderboard(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
        epoch: Option<i64>,
    ) -> Result<Connection<String, ModelStats>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first.unwrap_or(config.default_page_size).min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_targets;

        // Load all targets (any status) for assignment counts, and filled targets for win stats.
        // We need: target_id, cp_sequence_number, status, winning_model_id, scores, reward, data_size, model_ids_json
        type Row = (
            Vec<u8>,         // target_id
            i64,             // cp_sequence_number
            String,          // status
            Option<Vec<u8>>, // winning_model_id
            Option<f64>,     // winning_distance_score
            Option<f64>,     // winning_loss_score
            i64,             // reward_pool
            Option<i64>,     // winning_data_size
            String,          // model_ids_json
        );

        let mut query = soma_targets::table
            .select((
                soma_targets::target_id,
                soma_targets::cp_sequence_number,
                soma_targets::status,
                soma_targets::winning_model_id,
                soma_targets::winning_distance_score,
                soma_targets::winning_loss_score,
                soma_targets::reward_pool,
                soma_targets::winning_data_size,
                soma_targets::model_ids_json,
            ))
            .order((soma_targets::target_id.asc(), soma_targets::cp_sequence_number.desc()))
            .into_boxed();

        if let Some(e) = epoch {
            query = query.filter(soma_targets::epoch.eq(e));
        }

        let rows: Vec<Row> =
            query.load(conn.deref_mut()).await.map_err(|e| Error::new(e.to_string()))?;

        // Deduplicate by target_id (take most recent cp_sequence_number).
        // For each unique target:
        //   - Parse model_ids_json to count assignments per model
        //   - If status == "filled", aggregate win stats by winning_model_id
        let mut seen = std::collections::HashSet::new();
        // Value: (targets_won, dist_sum, dist_count, loss_sum, loss_count, total_reward, total_data_size, targets_assigned)
        let mut model_map: std::collections::HashMap<
            Vec<u8>,
            (i64, f64, i64, f64, i64, i64, i64, i64),
        > = std::collections::HashMap::new();

        for (tid, _cp, status, winning_model_id, dist, loss, rp, data_size, model_ids_json) in &rows
        {
            if !seen.insert(tid.clone()) {
                continue;
            }

            // Count assignments for every model listed in model_ids_json
            let assigned_ids: Vec<String> =
                serde_json::from_str(model_ids_json).unwrap_or_default();
            for hex_str in &assigned_ids {
                let hex = hex_str.strip_prefix("0x").unwrap_or(hex_str);
                if let Ok(mid) = hex::decode(hex) {
                    model_map.entry(mid).or_default().7 += 1;
                }
            }

            // Aggregate win stats for filled targets
            if status.eq_ignore_ascii_case("filled") {
                let Some(wm) = winning_model_id else {
                    continue;
                };
                let entry = model_map.entry(wm.clone()).or_default();
                entry.0 += 1; // targets_won
                if let Some(d) = dist {
                    entry.1 += d;
                    entry.2 += 1;
                }
                if let Some(l) = loss {
                    entry.3 += l;
                    entry.4 += 1;
                }
                entry.5 = entry.5.saturating_add(*rp);
                if let Some(ds) = data_size {
                    entry.6 = entry.6.saturating_add(*ds);
                }
            }
        }

        // Sort by targets_won desc
        let mut entries: Vec<_> = model_map.into_iter().collect();
        entries.sort_by(|a, b| b.1.0.cmp(&a.1.0));

        // Apply cursor pagination
        let after_addr: Option<Vec<u8>> = after
            .as_deref()
            .map(|s| hex::decode(s).map_err(|e| Error::new(format!("Invalid cursor: {e}"))))
            .transpose()?;

        let start_idx = match &after_addr {
            Some(addr) => entries.iter().position(|(k, _)| k == addr).map(|i| i + 1).unwrap_or(0),
            None => 0,
        };

        let page: Vec<_> = entries.into_iter().skip(start_idx).take(limit as usize + 1).collect();
        let has_next = page.len() as i64 > limit;
        let nodes: Vec<_> = page.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for (mid, (tw, ds, dc, ls, lc, tr, tds, ta)) in nodes {
            let cursor = hex::encode(&mid);
            connection.edges.push(Edge::new(
                cursor,
                ModelStats {
                    model_id: mid,
                    targets_won: tw,
                    targets_assigned: ta,
                    avg_distance_score: if dc > 0 { Some(ds / dc as f64) } else { None },
                    avg_loss_score: if lc > 0 { Some(ls / lc as f64) } else { None },
                    total_reward: tr,
                    total_data_size: tds,
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

/// Convert nested tuple rows into a Target.
type TargetRowA = (Vec<u8>, i64, i64, String, Option<Vec<u8>>, Option<Vec<u8>>, i64, i64, i32);
type TargetRowB = (
    Option<f64>,
    Option<f64>,
    Option<Vec<u8>>,
    Option<i64>,
    f64,
    String,
    Option<String>,
    Option<Vec<u8>>,
    Option<i64>,
);

fn target_from_row(a: TargetRowA, b: TargetRowB) -> Target {
    Target {
        target_id: a.0,
        cp_sequence_number: a.1,
        epoch: a.2,
        status: a.3,
        submitter: a.4,
        winning_model_id: a.5,
        reward_pool: a.6,
        bond_amount: a.7,
        report_count: a.8,
        winning_distance_score: b.0,
        winning_loss_score: b.1,
        winning_model_owner: b.2,
        fill_epoch: b.3,
        distance_threshold: b.4,
        model_ids_json: b.5,
        winning_data_url: b.6,
        winning_data_checksum: b.7,
        winning_data_size: b.8,
    }
}

// ---------------------------------------------------------------------------
// Model row types — nested tuples to stay within Diesel's 16-element limit
// ---------------------------------------------------------------------------

/// First 13 columns of soma_models.
type ModelRowA = (
    Vec<u8>,
    i64,
    String,
    Vec<u8>,
    i64,
    i64,
    i64,
    i64,
    i64,
    Vec<u8>,
    Option<i64>,
    Option<i64>,
    i64,
);

/// Middle 10 columns of soma_models.
type ModelRowB = (
    i64,
    i64,
    i64,
    i64,
    String,
    Option<String>,
    Option<Vec<u8>>,
    Option<i64>,
    Option<Vec<u8>>,
    bool,
);

/// Last 5 columns of soma_models (pending update fields).
type ModelRowC = (Option<String>, Option<Vec<u8>>, Option<i64>, Option<Vec<u8>>, Option<i64>);

fn model_select() -> (
    (
        indexer_alt_schema::schema::soma_models::model_id,
        indexer_alt_schema::schema::soma_models::epoch,
        indexer_alt_schema::schema::soma_models::status,
        indexer_alt_schema::schema::soma_models::owner,
        indexer_alt_schema::schema::soma_models::architecture_version,
        indexer_alt_schema::schema::soma_models::commit_epoch,
        indexer_alt_schema::schema::soma_models::stake,
        indexer_alt_schema::schema::soma_models::commission_rate,
        indexer_alt_schema::schema::soma_models::next_epoch_commission_rate,
        indexer_alt_schema::schema::soma_models::staking_pool_id,
        indexer_alt_schema::schema::soma_models::activation_epoch,
        indexer_alt_schema::schema::soma_models::deactivation_epoch,
        indexer_alt_schema::schema::soma_models::rewards_pool,
    ),
    (
        indexer_alt_schema::schema::soma_models::pool_token_balance,
        indexer_alt_schema::schema::soma_models::pending_stake,
        indexer_alt_schema::schema::soma_models::pending_total_soma_withdraw,
        indexer_alt_schema::schema::soma_models::pending_pool_token_withdraw,
        indexer_alt_schema::schema::soma_models::exchange_rates_json,
        indexer_alt_schema::schema::soma_models::manifest_url,
        indexer_alt_schema::schema::soma_models::manifest_checksum,
        indexer_alt_schema::schema::soma_models::manifest_size,
        indexer_alt_schema::schema::soma_models::weights_commitment,
        indexer_alt_schema::schema::soma_models::has_pending_update,
    ),
    (
        indexer_alt_schema::schema::soma_models::pending_manifest_url,
        indexer_alt_schema::schema::soma_models::pending_manifest_checksum,
        indexer_alt_schema::schema::soma_models::pending_manifest_size,
        indexer_alt_schema::schema::soma_models::pending_weights_commitment,
        indexer_alt_schema::schema::soma_models::pending_commit_epoch,
    ),
) {
    use indexer_alt_schema::schema::soma_models;
    (
        (
            soma_models::model_id,
            soma_models::epoch,
            soma_models::status,
            soma_models::owner,
            soma_models::architecture_version,
            soma_models::commit_epoch,
            soma_models::stake,
            soma_models::commission_rate,
            soma_models::next_epoch_commission_rate,
            soma_models::staking_pool_id,
            soma_models::activation_epoch,
            soma_models::deactivation_epoch,
            soma_models::rewards_pool,
        ),
        (
            soma_models::pool_token_balance,
            soma_models::pending_stake,
            soma_models::pending_total_soma_withdraw,
            soma_models::pending_pool_token_withdraw,
            soma_models::exchange_rates_json,
            soma_models::manifest_url,
            soma_models::manifest_checksum,
            soma_models::manifest_size,
            soma_models::weights_commitment,
            soma_models::has_pending_update,
        ),
        (
            soma_models::pending_manifest_url,
            soma_models::pending_manifest_checksum,
            soma_models::pending_manifest_size,
            soma_models::pending_weights_commitment,
            soma_models::pending_commit_epoch,
        ),
    )
}

fn model_from_row(a: ModelRowA, b: ModelRowB, c: ModelRowC) -> Model {
    Model {
        model_id: a.0,
        epoch: a.1,
        status: a.2,
        owner: a.3,
        architecture_version: a.4,
        commit_epoch: a.5,
        stake: a.6,
        commission_rate: a.7,
        next_epoch_commission_rate: a.8,
        staking_pool_id: a.9,
        activation_epoch: a.10,
        deactivation_epoch: a.11,
        rewards_pool: a.12,
        pool_token_balance: b.0,
        pending_stake: b.1,
        pending_total_soma_withdraw: b.2,
        pending_pool_token_withdraw: b.3,
        exchange_rates_json: b.4,
        manifest_url: b.5,
        manifest_checksum: b.6,
        manifest_size: b.7,
        weights_commitment: b.8,
        has_pending_update: b.9,
        pending_manifest_url: c.0,
        pending_manifest_checksum: c.1,
        pending_manifest_size: c.2,
        pending_weights_commitment: c.3,
        pending_commit_epoch: c.4,
    }
}
