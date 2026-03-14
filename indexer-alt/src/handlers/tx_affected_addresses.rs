// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::tx_affected_addresses;
use indexer_alt_schema::transactions::StoredTxAffectedAddress;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;

use crate::handlers::affected_addresses;
use crate::handlers::cp_sequence_numbers::tx_interval;

pub struct TxAffectedAddresses;

#[async_trait]
impl Processor for TxAffectedAddresses {
    const NAME: &'static str = "tx_affected_addresses";

    type Value = StoredTxAffectedAddress;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, .. } = checkpoint.as_ref();

        let mut values = Vec::new();
        let first_tx = summary.network_total_transactions as usize - transactions.len();

        for (i, tx) in transactions.iter().enumerate() {
            let tx_sequence_number = (first_tx + i) as i64;
            let sender = tx.transaction.sender();
            let recipients = affected_addresses(&tx.effects);

            let mut seen = BTreeSet::new();
            seen.insert(sender);

            let mut affected: Vec<StoredTxAffectedAddress> = vec![StoredTxAffectedAddress {
                tx_sequence_number,
                affected: sender.to_vec(),
                sender: sender.to_vec(),
            }];

            for addr in recipients {
                if seen.insert(addr) {
                    affected.push(StoredTxAffectedAddress {
                        tx_sequence_number,
                        affected: addr.to_vec(),
                        sender: sender.to_vec(),
                    });
                }
            }

            values.extend(affected);
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for TxAffectedAddresses {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(tx_affected_addresses::table)
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
        let filter = tx_affected_addresses::table.filter(
            tx_affected_addresses::tx_sequence_number.between(from_tx as i64, to_tx as i64 - 1),
        );

        Ok(diesel::delete(filter).execute(conn).await?)
    }
}
