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

use crate::api::types::address::Address;
use crate::api::types::checkpoint::Checkpoint;
use crate::api::types::epoch::Epoch;
use crate::api::types::model::Model;
use crate::api::types::object::Object as GqlObject;
use crate::api::types::reward::Reward;
use crate::api::types::service_config::ServiceConfig;
use crate::api::types::target::{Target, TargetFilter};
use crate::api::types::transaction::Transaction;
use crate::config::GraphQlConfig;
use crate::db::PgReader;

pub struct Query;

#[Object]
impl Query {
    /// The chain identifier (base58 hash of genesis checkpoint summary).
    async fn chain_identifier(&self, ctx: &Context<'_>) -> Result<String> {
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

        let seq = match sequence_number {
            Some(s) => s,
            None => {
                kv_checkpoints::table
                    .select(kv_checkpoints::sequence_number)
                    .order(kv_checkpoints::sequence_number.desc())
                    .first::<i64>(conn.deref_mut())
                    .await
                    .optional()
                    .map_err(|e| Error::new(e.to_string()))?
                    .ok_or_else(|| Error::new("No checkpoints found"))?
            }
        };

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

        let cp_info: Option<(i64, i64)> = cp_sequence_numbers::table
            .select((cp_sequence_numbers::epoch, cp_sequence_numbers::tx_lo))
            .filter(cp_sequence_numbers::cp_sequence_number.eq(seq))
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

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
    async fn transaction(
        &self,
        ctx: &Context<'_>,
        digest: String,
    ) -> Result<Option<Transaction>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        let digest_bytes = bs58::decode(&digest)
            .into_vec()
            .map_err(|e| Error::new(format!("Invalid digest: {e}")))?;

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

        let ver = match version {
            Some(v) => v,
            None => {
                obj_versions::table
                    .select(obj_versions::object_version)
                    .filter(obj_versions::object_id.eq(&id_bytes))
                    .order(obj_versions::object_version.desc())
                    .first::<i64>(conn.deref_mut())
                    .await
                    .optional()
                    .map_err(|e| Error::new(e.to_string()))?
                    .ok_or_else(|| Error::new("Object not found"))?
            }
        };

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

        let object_type = info
            .as_ref()
            .and_then(|i| i.module.as_ref().cloned());

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
        Ok(Address {
            address: addr_bytes,
        })
    }

    /// Look up an epoch by ID. Returns the latest if omitted.
    async fn epoch(&self, ctx: &Context<'_>, epoch_id: Option<i64>) -> Result<Option<Epoch>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::{kv_epoch_ends, kv_epoch_starts};

        let epoch_num = match epoch_id {
            Some(e) => e,
            None => {
                kv_epoch_starts::table
                    .select(kv_epoch_starts::epoch)
                    .order(kv_epoch_starts::epoch.desc())
                    .first::<i64>(conn.deref_mut())
                    .await
                    .optional()
                    .map_err(|e| Error::new(e.to_string()))?
                    .ok_or_else(|| Error::new("No epochs found"))?
            }
        };

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

        type TargetRow = (
            Vec<u8>, i64, i64, String, Option<Vec<u8>>, Option<Vec<u8>>,
            i64, i64, i32, Vec<u8>,
        );

        let stored: Option<TargetRow> = soma_targets::table
            .select((
                soma_targets::target_id,
                soma_targets::cp_sequence_number,
                soma_targets::epoch,
                soma_targets::status,
                soma_targets::submitter,
                soma_targets::winning_model_id,
                soma_targets::reward_pool,
                soma_targets::bond_amount,
                soma_targets::report_count,
                soma_targets::state_bcs,
            ))
            .filter(soma_targets::target_id.eq(&id_bytes))
            .order(soma_targets::cp_sequence_number.desc())
            .first(conn.deref_mut())
            .await
            .optional()
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(stored.map(|s| Target {
            target_id: s.0,
            cp_sequence_number: s.1,
            epoch: s.2,
            status: s.3,
            submitter: s.4,
            winning_model_id: s.5,
            reward_pool: s.6,
            bond_amount: s.7,
            report_count: s.8,
            state_bcs: s.9,
        }))
    }

    /// Query targets with pagination and optional filters.
    async fn targets(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
        filter: Option<TargetFilter>,
    ) -> Result<Connection<String, Target>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first
            .unwrap_or(config.default_page_size)
            .min(config.max_page_size) as i64;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_targets;

        let after_cp: Option<i64> = match &after {
            Some(cursor) => {
                let parts: Vec<&str> = cursor.splitn(2, ':').collect();
                if parts.len() != 2 {
                    return Err(Error::new("Invalid cursor format"));
                }
                let cp: i64 = parts[1]
                    .parse()
                    .map_err(|e| Error::new(format!("Invalid cursor cp: {e}")))?;
                Some(cp)
            }
            None => None,
        };

        type Row = (
            Vec<u8>, i64, i64, String, Option<Vec<u8>>, Option<Vec<u8>>,
            i64, i64, i32, Vec<u8>,
        );

        let mut query = soma_targets::table
            .select((
                soma_targets::target_id,
                soma_targets::cp_sequence_number,
                soma_targets::epoch,
                soma_targets::status,
                soma_targets::submitter,
                soma_targets::winning_model_id,
                soma_targets::reward_pool,
                soma_targets::bond_amount,
                soma_targets::report_count,
                soma_targets::state_bcs,
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
        }

        if let Some(acp) = after_cp {
            query = query.filter(soma_targets::cp_sequence_number.lt(acp));
        }

        let results: Vec<Row> = query
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let has_next = results.len() as i64 > limit;
        let nodes: Vec<_> = results.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for s in nodes {
            let cursor = format!("{}:{}", hex::encode(&s.0), s.1);
            connection.edges.push(Edge::new(
                cursor,
                Target {
                    target_id: s.0,
                    cp_sequence_number: s.1,
                    epoch: s.2,
                    status: s.3,
                    submitter: s.4,
                    winning_model_id: s.5,
                    reward_pool: s.6,
                    bond_amount: s.7,
                    report_count: s.8,
                    state_bcs: s.9,
                },
            ));
        }

        Ok(connection)
    }

    /// Query models with pagination. Returns models from the latest epoch by default.
    async fn models(
        &self,
        ctx: &Context<'_>,
        first: Option<i32>,
        after: Option<String>,
        epoch: Option<i64>,
    ) -> Result<Connection<String, Model>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let config: &GraphQlConfig = ctx.data()?;
        let limit = first
            .unwrap_or(config.default_page_size)
            .min(config.max_page_size) as i64;
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

        type Row = (
            Vec<u8>, i64, String, Vec<u8>, i64, i64, i64, i64, bool, Vec<u8>,
        );

        let mut query = soma_models::table
            .select((
                soma_models::model_id,
                soma_models::epoch,
                soma_models::status,
                soma_models::owner,
                soma_models::architecture_version,
                soma_models::commit_epoch,
                soma_models::stake,
                soma_models::commission_rate,
                soma_models::has_embedding,
                soma_models::state_bcs,
            ))
            .filter(soma_models::epoch.eq(epoch_num))
            .order(soma_models::model_id.asc())
            .limit(limit + 1)
            .into_boxed();

        if let Some(ref aid) = after_id {
            query = query.filter(soma_models::model_id.gt(aid));
        }

        let results: Vec<Row> = query
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        let has_next = results.len() as i64 > limit;
        let nodes: Vec<_> = results.into_iter().take(limit as usize).collect();
        let has_previous = after.is_some();

        let mut connection = Connection::new(has_previous, has_next);
        for s in nodes {
            let cursor = hex::encode(&s.0);
            connection.edges.push(Edge::new(
                cursor,
                Model {
                    model_id: s.0,
                    epoch: s.1,
                    status: s.2,
                    owner: s.3,
                    architecture_version: s.4,
                    commit_epoch: s.5,
                    stake: s.6,
                    commission_rate: s.7,
                    has_embedding: s.8,
                    state_bcs: s.9,
                },
            ));
        }

        Ok(connection)
    }

    /// Query rewards for a specific epoch and optionally a specific target.
    async fn rewards(
        &self,
        ctx: &Context<'_>,
        epoch: i64,
        target_id: Option<String>,
    ) -> Result<Vec<Reward>> {
        let pg: &Arc<PgReader> = ctx.data()?;
        let mut conn = pg.connect().await?;

        use indexer_alt_schema::schema::soma_rewards;

        type Row = (Vec<u8>, i64, i64, Vec<u8>, Vec<u8>);

        let mut query = soma_rewards::table
            .select((
                soma_rewards::target_id,
                soma_rewards::cp_sequence_number,
                soma_rewards::epoch,
                soma_rewards::tx_digest,
                soma_rewards::balance_changes_bcs,
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

        let results: Vec<Row> = query
            .load(conn.deref_mut())
            .await
            .map_err(|e| Error::new(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|s| Reward {
                target_id: s.0,
                cp_sequence_number: s.1,
                epoch: s.2,
                tx_digest: s.3,
                balance_changes_bcs: s.4,
            })
            .collect())
    }
}
