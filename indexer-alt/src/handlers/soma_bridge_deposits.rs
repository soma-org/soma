// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-tx materialization of `BridgeDeposit` transactions.
//!
//! One row per BridgeDeposit, capturing recipient + amount + nonce + the
//! Base-side `eth_tx_hash`. The latter is the only place this proof is
//! retained; `soma_balance_deltas` is per-cp-aggregated and drops it.
//! Downstream consumers join here when they need to correlate Soma
//! recipients with their L1 funding source.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_bridge_deposits;
use indexer_alt_schema::soma::StoredBridgeDeposit;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::transaction::TransactionKind;

pub struct SomaBridgeDeposits;

#[async_trait]
impl Processor for SomaBridgeDeposits {
    const NAME: &'static str = "soma_bridge_deposits";

    type Value = StoredBridgeDeposit;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, .. } = checkpoint.as_ref();
        let cp = summary.sequence_number as i64;
        let timestamp_ms = summary.timestamp_ms as i64;
        let first_tx = summary.network_total_transactions as usize - transactions.len();

        let mut out = Vec::new();
        for (i, tx) in transactions.iter().enumerate() {
            let TransactionKind::BridgeDeposit(args) = tx.transaction.kind() else {
                continue;
            };
            out.push(StoredBridgeDeposit {
                tx_sequence_number: (first_tx + i) as i64,
                cp_sequence_number: cp,
                recipient: args.recipient.to_vec(),
                amount: args.amount as i64,
                nonce: args.nonce as i64,
                eth_tx_hash: args.eth_tx_hash.to_vec(),
                timestamp_ms,
            });
        }
        Ok(out)
    }
}

#[async_trait]
impl Handler for SomaBridgeDeposits {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_bridge_deposits::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }

    async fn prune<'a>(
        &self,
        _from: u64,
        _to_exclusive: u64,
        _conn: &mut Connection<'a>,
    ) -> Result<usize> {
        // Bridge deposits are forensic data — never pruned.
        Ok(0)
    }
}
