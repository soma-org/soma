// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Denormalized per-Settle row: voucher cumulative_* + channel
//! model_id + payee in one place. Joins data that already exists in
//! `soma_channels` and `soma_channel_events`, but in a form that
//! answers "how many tokens of model X did provider Y deliver today"
//! with a single table scan.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_inference_settlements;
use indexer_alt_schema::soma::StoredInferenceSettlement;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::object::Object;
use types::transaction::TransactionKind;

pub struct SomaInferenceSettlements;

#[async_trait]
impl Processor for SomaInferenceSettlements {
    const NAME: &'static str = "soma_inference_settlements";

    type Value = StoredInferenceSettlement;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, object_set, .. } = checkpoint.as_ref();
        let cp = summary.sequence_number as i64;
        let timestamp_ms = summary.timestamp_ms as i64;
        let first_tx = summary.network_total_transactions as usize - transactions.len();

        let mut out = Vec::new();
        for (i, tx) in transactions.iter().enumerate() {
            let TransactionKind::Settle(args) = tx.transaction.kind() else {
                continue;
            };

            let tx_seq = (first_tx + i) as i64;

            // Pre-state Channel carries the snapshotted model_id, payer,
            // payee, and the prior `settled_amount` (for the delta).
            let pre: Vec<&Object> = tx.input_objects(object_set).collect();
            let Some((id, pre_chan)) = pre.iter().find_map(|o| o.as_channel().map(|c| (o.id(), c)))
            else {
                continue;
            };
            // Post-state has the new settled_amount; fall back to pre if
            // it wasn't mutated (shouldn't happen for Settle, but
            // defensive).
            let post: Vec<&Object> = tx.output_objects(object_set).collect();
            let post_settled = post
                .iter()
                .find_map(|o| o.as_channel().map(|c| c.settled_amount()))
                .unwrap_or(pre_chan.settled_amount());
            let delta_amount = post_settled.saturating_sub(pre_chan.settled_amount()) as i64;

            out.push(StoredInferenceSettlement {
                tx_sequence_number: tx_seq,
                cp_sequence_number: cp,
                channel_id: id.to_vec(),
                payer: pre_chan.payer().to_vec(),
                payee: pre_chan.payee().to_vec(),
                model_id: pre_chan.model_id().to_string(),
                cumulative_amount: args.cumulative_amount as i64,
                cumulative_prompt_tokens: args.cumulative_prompt_tokens as i64,
                cumulative_completion_tokens: args.cumulative_completion_tokens as i64,
                cumulative_cache_read_tokens: args.cumulative_cache_read_tokens as i64,
                cumulative_cache_write_tokens: args.cumulative_cache_write_tokens as i64,
                cumulative_requests: args.cumulative_requests as i64,
                delta_amount,
                timestamp_ms,
            });
        }

        Ok(out)
    }
}

#[async_trait]
impl Handler for SomaInferenceSettlements {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10_000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_inference_settlements::table)
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
        // Tier C — never pruned. Feeds the realized-price oracle and
        // long-window aggregates.
        Ok(0)
    }
}
