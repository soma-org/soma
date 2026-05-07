// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! On-chain provider registry mirror.
//!
//! INSERT on RegisterProvider; UPSERT on UpdateProvider (the chain
//! treats the latter as a heartbeat that always bumps
//! `registered_at_ms`).

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_providers;
use indexer_alt_schema::soma::StoredProvider;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::object::Object;
use types::transaction::TransactionKind;

pub struct SomaProviders;

#[async_trait]
impl Processor for SomaProviders {
    const NAME: &'static str = "soma_providers";

    type Value = StoredProvider;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, object_set, .. } = checkpoint.as_ref();
        let cp = summary.sequence_number as i64;

        let mut out = Vec::new();
        for tx in transactions {
            let kind = tx.transaction.kind();
            if !kind.is_provider_tx() {
                continue;
            }
            let post: Vec<&Object> = tx.output_objects(object_set).collect();
            let provider = post.iter().find_map(|o| o.as_provider());
            let Some(p) = provider else { continue };
            out.push(StoredProvider {
                address: p.address.to_vec(),
                endpoint: p.endpoint,
                registered_at_ms: p.registered_at_ms as i64,
                last_update_cp: cp,
            });
        }
        Ok(out)
    }
}

#[async_trait]
impl Handler for SomaProviders {
    const MIN_EAGER_ROWS: usize = 10;
    const MAX_PENDING_ROWS: usize = 1_000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_providers::table)
            .values(values)
            .on_conflict(soma_providers::address)
            .do_update()
            .set((
                soma_providers::endpoint.eq(diesel::dsl::sql::<diesel::sql_types::Text>(
                    "EXCLUDED.endpoint",
                )),
                soma_providers::registered_at_ms.eq(
                    diesel::dsl::sql::<diesel::sql_types::Int8>("EXCLUDED.registered_at_ms"),
                ),
                soma_providers::last_update_cp.eq(diesel::dsl::sql::<diesel::sql_types::Int8>(
                    "EXCLUDED.last_update_cp",
                )),
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
