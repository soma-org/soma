// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-(provider, model_id) offering mirror.
//!
//! UPSERT on `RegisterOffering` / `UpdateOffering` / `DeactivateOffering`.
//! One row per (provider, model_id); reactivation via UpdateOffering
//! flips `active` back to TRUE. The handler reads the post-state of
//! every offering tx's output objects and lets the executor's
//! invariants (only the owner can mutate, model_id ∈ ModelRegistry)
//! filter ahead of time — no validation here.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_offerings;
use indexer_alt_schema::soma::StoredOffering;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::object::Object;

pub struct SomaOfferings;

#[async_trait]
impl Processor for SomaOfferings {
    const NAME: &'static str = "soma_offerings";

    type Value = StoredOffering;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, object_set, .. } = checkpoint.as_ref();
        let cp = summary.sequence_number as i64;
        let timestamp_ms = summary.timestamp_ms as i64;

        let mut out = Vec::new();
        for tx in transactions {
            if !tx.transaction.kind().is_offering_tx() {
                continue;
            }
            // Each offering tx mutates exactly one Offering object — find
            // the post-state row and emit a single StoredOffering.
            let post: Vec<&Object> = tx.output_objects(object_set).collect();
            let Some(o) = post.iter().find_map(|o| o.as_offering()) else {
                continue;
            };
            out.push(StoredOffering {
                provider: o.provider().to_vec(),
                model_id: o.model_id().to_string(),
                prompt_micros_per_1k: o.prompt_micros_per_1k() as i64,
                completion_micros_per_1k: o.completion_micros_per_1k() as i64,
                cache_read_micros_per_1k: o.cache_read_micros_per_1k() as i64,
                cache_write_micros_per_1k: o.cache_write_micros_per_1k() as i64,
                request_micros: o.request_micros() as i64,
                ttft_bound_ms: o.ttft_bound_ms() as i32,
                ttot_bound_ms: o.ttot_bound_ms() as i32,
                active: o.active(),
                updated_at_cp: cp,
                // Prefer the on-chain `updated_at_ms` (clock-derived,
                // signed into the row) over the checkpoint timestamp.
                updated_at_ms: o.updated_at_ms() as i64,
            });
            let _ = timestamp_ms; // explicit unused — see above
        }
        Ok(out)
    }
}

#[async_trait]
impl Handler for SomaOfferings {
    const MIN_EAGER_ROWS: usize = 10;
    const MAX_PENDING_ROWS: usize = 1_000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_offerings::table)
            .values(values)
            .on_conflict((soma_offerings::provider, soma_offerings::model_id))
            .do_update()
            .set((
                soma_offerings::prompt_micros_per_1k.eq(
                    diesel::dsl::sql::<diesel::sql_types::Int8>("EXCLUDED.prompt_micros_per_1k"),
                ),
                soma_offerings::completion_micros_per_1k.eq(diesel::dsl::sql::<
                    diesel::sql_types::Int8,
                >(
                    "EXCLUDED.completion_micros_per_1k"
                )),
                soma_offerings::cache_read_micros_per_1k.eq(diesel::dsl::sql::<
                    diesel::sql_types::Int8,
                >(
                    "EXCLUDED.cache_read_micros_per_1k"
                )),
                soma_offerings::cache_write_micros_per_1k.eq(diesel::dsl::sql::<
                    diesel::sql_types::Int8,
                >(
                    "EXCLUDED.cache_write_micros_per_1k",
                )),
                soma_offerings::request_micros
                    .eq(diesel::dsl::sql::<diesel::sql_types::Int8>("EXCLUDED.request_micros")),
                soma_offerings::ttft_bound_ms
                    .eq(diesel::dsl::sql::<diesel::sql_types::Int4>("EXCLUDED.ttft_bound_ms")),
                soma_offerings::ttot_bound_ms
                    .eq(diesel::dsl::sql::<diesel::sql_types::Int4>("EXCLUDED.ttot_bound_ms")),
                soma_offerings::active
                    .eq(diesel::dsl::sql::<diesel::sql_types::Bool>("EXCLUDED.active")),
                soma_offerings::updated_at_cp
                    .eq(diesel::dsl::sql::<diesel::sql_types::Int8>("EXCLUDED.updated_at_cp")),
                soma_offerings::updated_at_ms
                    .eq(diesel::dsl::sql::<diesel::sql_types::Int8>("EXCLUDED.updated_at_ms")),
            ))
            .execute(conn)
            .await?)
    }

    async fn prune<'a>(
        &self,
        _from: u64,
        _to_exclusive: u64,
        _conn: &mut Connection<'a>,
    ) -> Result<usize> {
        Ok(0)
    }
}
