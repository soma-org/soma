// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-channel state mirror, sourced from on-chain `Channel` objects.
//!
//! For each transaction in the checkpoint:
//!
//! | tx kind             | row op                                                           |
//! |---------------------|------------------------------------------------------------------|
//! | OpenChannel         | INSERT row from output `Channel`                                 |
//! | Settle              | UPSERT updated `(deposit, settled_amount, last_update_cp)`       |
//! | TopUp               | UPSERT updated deposit + clear `close_requested_at_ms`           |
//! | RequestClose        | UPSERT `close_requested_at_ms`, status=closing                   |
//! | WithdrawAfterTimeout| UPSERT status=withdrawn, deposit=0; row preserved (chain deletes |
//! |                     |   the object but we keep the record for client recovery and rep) |
//!
//! Status discriminator (i16):
//!   * 0 = open
//!   * 1 = closing
//!   * 2 = withdrawn

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_channels;
use indexer_alt_schema::soma::StoredChannel;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::channel::Channel;
use types::full_checkpoint_content::Checkpoint;
use types::object::{Object, ObjectID};
use types::transaction::TransactionKind;

const STATUS_OPEN: i16 = 0;
const STATUS_CLOSING: i16 = 1;
const STATUS_WITHDRAWN: i16 = 2;

pub struct SomaChannels;

fn coin_type_label(c: types::object::CoinType) -> &'static str {
    match c {
        types::object::CoinType::Soma => "SOMA",
        types::object::CoinType::Usdc => "USDC",
    }
}

fn channel_to_stored(
    id: ObjectID,
    chan: &Channel,
    opened_at_cp: i64,
    opened_tx_digest: Vec<u8>,
    last_update_cp: i64,
    status: i16,
) -> StoredChannel {
    StoredChannel {
        channel_id: id.to_vec(),
        payer: chan.payer().to_vec(),
        payee: chan.payee().to_vec(),
        authorized_signer: chan.authorized_signer().to_vec(),
        token: coin_type_label(chan.token()).to_string(),
        deposit: chan.deposit() as i64,
        settled_amount: chan.settled_amount() as i64,
        close_requested_at_ms: chan.close_requested_at_ms().map(|v| v as i64),
        status,
        opened_at_cp,
        opened_tx_digest,
        last_update_cp,
        // Offering snapshot from the on-chain ChannelV1 record.
        model_id: chan.model_id().to_string(),
        prompt_micros_per_1k: chan.prompt_micros_per_1k() as i64,
        completion_micros_per_1k: chan.completion_micros_per_1k() as i64,
        cache_read_micros_per_1k: chan.cache_read_micros_per_1k() as i64,
        cache_write_micros_per_1k: chan.cache_write_micros_per_1k() as i64,
        request_micros: chan.request_micros() as i64,
        ttft_bound_ms: chan.ttft_bound_ms() as i32,
        ttot_bound_ms: chan.ttot_bound_ms() as i32,
    }
}

#[async_trait]
impl Processor for SomaChannels {
    const NAME: &'static str = "soma_channels";

    type Value = StoredChannel;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, object_set, .. } = checkpoint.as_ref();
        let cp = summary.sequence_number as i64;

        let mut out = Vec::new();
        for tx in transactions {
            let kind = tx.transaction.kind();
            if !kind.is_channel_tx() {
                continue;
            }

            let tx_digest = tx.transaction.digest().inner().to_vec();

            // Pre-state Channel (input) and post-state Channel (output)
            // pulled directly from the checkpoint's ObjectSet via the
            // effects' object_changes. This is the same pattern
            // soma_validators uses to read SystemState.
            let pre: Vec<&Object> = tx.input_objects(object_set).collect();
            let post: Vec<&Object> = tx.output_objects(object_set).collect();

            let pre_chan = pre.iter().find_map(|o| o.as_channel().map(|c| (o.id(), c)));
            let post_chan = post.iter().find_map(|o| o.as_channel().map(|c| (o.id(), c)));

            match kind {
                TransactionKind::OpenChannel(_) => {
                    if let Some((id, chan)) = post_chan {
                        out.push(channel_to_stored(
                            id,
                            &chan,
                            cp,
                            tx_digest.clone(),
                            cp,
                            STATUS_OPEN,
                        ));
                    }
                }
                TransactionKind::Settle(_) | TransactionKind::TopUp(_) => {
                    if let Some((id, chan)) = post_chan {
                        // For UPSERT we need a full row. opened_at_cp /
                        // opened_tx_digest are immutable post-INSERT —
                        // the Postgres `do_update` excludes them, so
                        // the values we pass here only matter on
                        // (impossible) first-row INSERT. Use the
                        // current checkpoint as a fallback to keep
                        // the column NOT NULL.
                        let status = if chan.close_requested_at_ms().is_some() {
                            STATUS_CLOSING
                        } else {
                            STATUS_OPEN
                        };
                        out.push(channel_to_stored(id, &chan, cp, tx_digest.clone(), cp, status));
                    }
                }
                TransactionKind::RequestClose(_) => {
                    if let Some((id, chan)) = post_chan {
                        out.push(channel_to_stored(
                            id,
                            &chan,
                            cp,
                            tx_digest.clone(),
                            cp,
                            STATUS_CLOSING,
                        ));
                    }
                }
                TransactionKind::WithdrawAfterTimeout(args) => {
                    // Object is deleted on this tx — `post` won't
                    // contain it. Use the pre-state to fill the
                    // immutable identity columns and override the
                    // mutating ones.
                    let id = args.channel_id;
                    let chan =
                        pre.iter().find_map(|o| if o.id() == id { o.as_channel() } else { None });
                    if let Some(chan) = chan {
                        // Manual: deposit=0, status=withdrawn,
                        // close_requested_at_ms preserved (the timer
                        // ran out — useful for "when did this end").
                        // Offering-snapshot columns: preserve so
                        // downstream queries can still attribute the
                        // closed channel to its model.
                        out.push(StoredChannel {
                            channel_id: id.to_vec(),
                            payer: chan.payer().to_vec(),
                            payee: chan.payee().to_vec(),
                            authorized_signer: chan.authorized_signer().to_vec(),
                            token: coin_type_label(chan.token()).to_string(),
                            deposit: 0,
                            settled_amount: chan.settled_amount() as i64,
                            close_requested_at_ms: chan.close_requested_at_ms().map(|v| v as i64),
                            status: STATUS_WITHDRAWN,
                            opened_at_cp: cp,
                            opened_tx_digest: tx_digest.clone(),
                            last_update_cp: cp,
                            model_id: chan.model_id().to_string(),
                            prompt_micros_per_1k: chan.prompt_micros_per_1k() as i64,
                            completion_micros_per_1k: chan.completion_micros_per_1k() as i64,
                            cache_read_micros_per_1k: chan.cache_read_micros_per_1k() as i64,
                            cache_write_micros_per_1k: chan.cache_write_micros_per_1k() as i64,
                            request_micros: chan.request_micros() as i64,
                            ttft_bound_ms: chan.ttft_bound_ms() as i32,
                            ttot_bound_ms: chan.ttot_bound_ms() as i32,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(out)
    }
}

#[async_trait]
impl Handler for SomaChannels {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10_000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        // Dedup by `channel_id` and keep the latest write per channel
        // within this batch. The framework can hand us multiple rows
        // for the same channel when several channel txs land in
        // adjacent cps inside one committer batch (e.g. a TopUp +
        // RequestClose on the same channel a few cps apart). PG's
        // `INSERT … ON CONFLICT … DO UPDATE` errors with "command
        // cannot affect row a second time" when two rows in the
        // same statement target the same conflict-key; dedup before
        // shipping so the UPSERT only sees one row per channel_id.
        // "Latest" = highest `last_update_cp`; ties broken by the
        // input order (the processor emits in tx order within a cp).
        let mut latest: std::collections::HashMap<Vec<u8>, &Self::Value> =
            std::collections::HashMap::with_capacity(values.len());
        for v in values {
            latest
                .entry(v.channel_id.clone())
                .and_modify(|prev| {
                    if v.last_update_cp >= prev.last_update_cp {
                        *prev = v;
                    }
                })
                .or_insert(v);
        }
        let deduped: Vec<Self::Value> = latest.into_values().cloned().collect();
        if deduped.is_empty() {
            return Ok(0);
        }

        // Standard Postgres ON CONFLICT update of the mutating columns.
        // `opened_at_cp`/`opened_tx_digest`/`payer`/`payee`/etc. stay at
        // their INSERT values; the mutating columns get overwritten on
        // every channel tx.
        Ok(diesel::insert_into(soma_channels::table)
            .values(&deduped)
            .on_conflict(soma_channels::channel_id)
            .do_update()
            .set((
                soma_channels::deposit
                    .eq(diesel::dsl::sql::<diesel::sql_types::Int8>("EXCLUDED.deposit")),
                soma_channels::settled_amount
                    .eq(diesel::dsl::sql::<diesel::sql_types::Int8>("EXCLUDED.settled_amount")),
                soma_channels::close_requested_at_ms.eq(diesel::dsl::sql::<
                    diesel::sql_types::Nullable<diesel::sql_types::Int8>,
                >(
                    "EXCLUDED.close_requested_at_ms"
                )),
                soma_channels::status
                    .eq(diesel::dsl::sql::<diesel::sql_types::Int2>("EXCLUDED.status")),
                soma_channels::last_update_cp
                    .eq(diesel::dsl::sql::<diesel::sql_types::Int8>("EXCLUDED.last_update_cp")),
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
        // Channels are recovery state — never prune.
        Ok(0)
    }
}
