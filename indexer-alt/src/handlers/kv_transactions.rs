// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use indexer_alt_schema::schema::kv_transactions;
use indexer_alt_schema::transactions::StoredTransaction;

pub struct KvTransactions;

#[async_trait]
impl Processor for KvTransactions {
    const NAME: &'static str = "kv_transactions";

    type Value = StoredTransaction;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint {
            transactions,
            summary,
            ..
        } = checkpoint.as_ref();

        let cp_sequence_number = summary.sequence_number as i64;
        let timestamp_ms = summary.timestamp_ms as i64;

        let mut values = Vec::with_capacity(transactions.len());
        for tx in transactions {
            let tx_digest = tx.transaction.digest().inner().to_vec();

            let raw_transaction =
                bcs::to_bytes(&tx.transaction).context("Serializing transaction")?;

            let raw_effects =
                bcs::to_bytes(&tx.effects).context("Serializing effects")?;

            // Soma has no events, store empty bytes
            let events = vec![];

            let user_signatures =
                bcs::to_bytes(&tx.signatures).context("Serializing user signatures")?;

            values.push(StoredTransaction {
                tx_digest,
                cp_sequence_number,
                timestamp_ms,
                raw_transaction,
                raw_effects,
                events,
                user_signatures,
            });
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for KvTransactions {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(kv_transactions::table)
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
        let filter = kv_transactions::table.filter(
            kv_transactions::cp_sequence_number.between(from as i64, to_exclusive as i64 - 1),
        );

        Ok(diesel::delete(filter).execute(conn).await?)
    }
}
