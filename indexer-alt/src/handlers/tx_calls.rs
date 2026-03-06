// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Soma does not have Move function calls (ProgrammableTransaction / MoveCall).
//! This handler exists for schema compatibility but always produces empty results.

use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use indexer_alt_schema::schema::tx_calls;
use indexer_alt_schema::transactions::StoredTxCalls;

use crate::handlers::cp_sequence_numbers::tx_interval;

pub struct TxCalls;

#[async_trait]
impl Processor for TxCalls {
    const NAME: &'static str = "tx_calls";

    type Value = StoredTxCalls;

    async fn process(&self, _checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        // Soma does not have Move function calls, so this pipeline always produces no rows.
        Ok(vec![])
    }
}

#[async_trait]
impl Handler for TxCalls {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(tx_calls::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }

    async fn prune<'a>(
        &self,
        from: u64,
        to_exclusive: u64,
        conn: &mut Connection<'a>,
    ) -> Result<usize> {
        let Range {
            start: from_tx,
            end: to_tx,
        } = tx_interval(conn, from..to_exclusive).await?;
        let filter = tx_calls::table
            .filter(tx_calls::tx_sequence_number.between(from_tx as i64, to_tx as i64 - 1));

        Ok(diesel::delete(filter).execute(conn).await?)
    }
}
