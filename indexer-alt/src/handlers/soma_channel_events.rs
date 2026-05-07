// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Append-only event log for payment-channel ops.
//!
//! One row per channel transaction. Powers reputation aggregates
//! (volume_settled_30d, distinct_buyers_30d, channel_renewal_rate,
//! mean_channel_age) and the per-channel history view used by clients
//! reconstructing state after losing local files.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_channel_events;
use indexer_alt_schema::soma::StoredChannelEvent;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::object::Object;
use types::transaction::TransactionKind;

pub struct SomaChannelEvents;

#[async_trait]
impl Processor for SomaChannelEvents {
    const NAME: &'static str = "soma_channel_events";

    type Value = StoredChannelEvent;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, object_set, .. } = checkpoint.as_ref();
        let cp = summary.sequence_number as i64;
        let timestamp_ms = summary.timestamp_ms as i64;
        let first_tx = summary.network_total_transactions as usize - transactions.len();

        let mut out = Vec::new();
        for (i, tx) in transactions.iter().enumerate() {
            let kind = tx.transaction.kind();
            if !kind.is_channel_tx() {
                continue;
            }

            let tx_seq = (first_tx + i) as i64;
            let pre: Vec<&Object> = tx.input_objects(object_set).collect();
            let post: Vec<&Object> = tx.output_objects(object_set).collect();
            let pre_chan = pre.iter().find_map(|o| o.as_channel().map(|c| (o.id(), c)));
            let post_chan = post.iter().find_map(|o| o.as_channel().map(|c| (o.id(), c)));

            let (channel_id, kind_label, delta) = match kind {
                TransactionKind::OpenChannel(_) => {
                    let Some((id, chan)) = post_chan else { continue };
                    (id, "open", chan.deposit as i64)
                }
                TransactionKind::Settle(_) => {
                    let Some((id, post_c)) = post_chan else { continue };
                    let pre_settled = pre_chan
                        .as_ref()
                        .map(|(_, c)| c.settled_amount)
                        .unwrap_or(0);
                    let delta = post_c.settled_amount.saturating_sub(pre_settled) as i64;
                    (id, "settle", delta)
                }
                TransactionKind::TopUp(args) => {
                    let id = args.channel_id;
                    let pre_dep =
                        pre_chan.as_ref().map(|(_, c)| c.deposit).unwrap_or(0);
                    let post_dep = post_chan.as_ref().map(|(_, c)| c.deposit).unwrap_or(0);
                    let delta = post_dep.saturating_sub(pre_dep) as i64;
                    (id, "top_up", delta)
                }
                TransactionKind::RequestClose(args) => (args.channel_id, "request_close", 0),
                TransactionKind::WithdrawAfterTimeout(args) => {
                    (args.channel_id, "withdraw", 0)
                }
                _ => continue,
            };

            out.push(StoredChannelEvent {
                tx_sequence_number: tx_seq,
                cp_sequence_number: cp,
                channel_id: channel_id.to_vec(),
                kind: kind_label.to_string(),
                delta,
                timestamp_ms,
            });
        }

        Ok(out)
    }
}

#[async_trait]
impl Handler for SomaChannelEvents {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10_000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_channel_events::table)
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
        // History — never prune. Reputation aggregates need the tail.
        Ok(0)
    }
}
