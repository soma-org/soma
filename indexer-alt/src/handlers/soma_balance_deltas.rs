// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-checkpoint balance-delta indexer.
//!
//! Reads `effects.balance_events()` straight off each transaction in
//! the checkpoint, aggregates per `(owner, coin_type)`, and emits one
//! row per touched key into `soma_balance_deltas`.
//!
//! Current balance = `SUM(delta) WHERE owner = ? AND coin_type = ?`.
//! GraphQL and any downstream consumer reads through that aggregate;
//! the materialized table is the single source of truth for indexer-
//! driven balance queries.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_balance_deltas;
use indexer_alt_schema::soma::StoredBalanceDelta;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::base::SomaAddress;
use types::effects::TransactionEffectsAPI;
use types::full_checkpoint_content::Checkpoint;
use types::object::CoinType;

pub struct SomaBalanceDeltas;

#[async_trait]
impl Processor for SomaBalanceDeltas {
    const NAME: &'static str = "soma_balance_deltas";

    type Value = StoredBalanceDelta;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, .. } = checkpoint.as_ref();
        let cp_sequence_number = summary.sequence_number as i64;

        // Aggregate every BalanceEvent in the checkpoint by
        // `(owner, coin_type)`. Multiple txs touching the same key
        // collapse to one row — settlement does the same math, so
        // the SUM-of-deltas across cps matches the on-chain
        // accumulator's running balance modulo timing.
        let mut net: BTreeMap<(SomaAddress, CoinType), i128> = BTreeMap::new();
        for tx in transactions {
            for ev in tx.effects.balance_events() {
                *net.entry((ev.owner(), ev.coin_type())).or_insert(0) += ev.signed_delta();
            }
        }

        let mut values = Vec::with_capacity(net.len());
        for ((owner, coin_type), delta) in net {
            if delta == 0 {
                continue;
            }
            // Saturate at i64 bounds defensively; per-cp deltas are
            // far below this in any realistic supply, so a saturated
            // value indicates a bug worth surfacing rather than
            // silently truncating.
            let delta_i64 = if delta > i64::MAX as i128 {
                i64::MAX
            } else if delta < i64::MIN as i128 {
                i64::MIN
            } else {
                delta as i64
            };
            values.push(StoredBalanceDelta {
                owner: owner.to_vec(),
                coin_type: coin_type.to_string(),
                cp_sequence_number,
                delta: delta_i64,
            });
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaBalanceDeltas {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10_000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_balance_deltas::table)
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
        // Pruning is keyed directly by `cp_sequence_number` — no need
        // for the tx_interval translation other handlers do.
        let filter = soma_balance_deltas::table.filter(
            soma_balance_deltas::cp_sequence_number.between(from as i64, to_exclusive as i64 - 1),
        );
        Ok(diesel::delete(filter).execute(conn).await?)
    }
}
