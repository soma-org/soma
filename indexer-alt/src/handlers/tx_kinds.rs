// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

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
use indexer_alt_schema::schema::tx_kinds;
use indexer_alt_schema::transactions::{StoredKind, StoredTxKind};

use crate::handlers::cp_sequence_numbers::tx_interval;

pub struct TxKinds;

#[async_trait]
impl Processor for TxKinds {
    const NAME: &'static str = "tx_kinds";

    type Value = StoredTxKind;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint {
            transactions,
            summary,
            ..
        } = checkpoint.as_ref();

        let mut values = Vec::new();
        let first_tx = summary.network_total_transactions as usize - transactions.len();

        for (i, tx) in transactions.iter().enumerate() {
            let tx_sequence_number = (first_tx + i) as i64;
            let tx_kind = if tx.transaction.is_system_tx() {
                StoredKind::SystemTransaction
            } else {
                StoredKind::ProgrammableTransaction
            };

            values.push(StoredTxKind {
                tx_sequence_number,
                tx_kind,
            });
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for TxKinds {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(tx_kinds::table)
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
        let filter = tx_kinds::table
            .filter(tx_kinds::tx_sequence_number.between(from_tx as i64, to_tx as i64 - 1));

        Ok(diesel::delete(filter).execute(conn).await?)
    }
}
