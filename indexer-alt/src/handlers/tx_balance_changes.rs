// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::tx_balance_changes;
use indexer_alt_schema::transactions::{BalanceChange, StoredTxBalanceChange};
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::balance_change::derive_balance_changes_2;
use types::full_checkpoint_content::Checkpoint;

use crate::handlers::cp_sequence_numbers::tx_interval;

pub struct TxBalanceChanges;

#[async_trait]
impl Processor for TxBalanceChanges {
    const NAME: &'static str = "tx_balance_changes";

    type Value = StoredTxBalanceChange;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, object_set, .. } = checkpoint.as_ref();

        let mut values = Vec::new();
        let first_tx = summary.network_total_transactions as usize - transactions.len();

        for (i, tx) in transactions.iter().enumerate() {
            let tx_sequence_number = (first_tx + i) as i64;

            let changes = derive_balance_changes_2(&tx.effects, object_set);

            // Convert to the schema's BalanceChange format
            let stored_changes: Vec<BalanceChange> = changes
                .into_iter()
                .map(|c| BalanceChange::V1 {
                    owner: c.address.to_vec(),
                    // Soma has a single native coin type
                    coin_type: "SOMA".to_string(),
                    amount: c.amount,
                })
                .collect();

            let balance_changes =
                bcs::to_bytes(&stored_changes).context("Serializing balance changes")?;

            values.push(StoredTxBalanceChange { tx_sequence_number, balance_changes });
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for TxBalanceChanges {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(tx_balance_changes::table)
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
        let Range { start: from_tx, end: to_tx } = tx_interval(conn, from..to_exclusive).await?;
        let filter = tx_balance_changes::table.filter(
            tx_balance_changes::tx_sequence_number.between(from_tx as i64, to_tx as i64 - 1),
        );

        Ok(diesel::delete(filter).execute(conn).await?)
    }
}
