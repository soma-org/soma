// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Channel ratings mirror.
//!
//! UPSERT on every `RateChannel` tx (latest-wins). Powers the
//! volume-weighted `mean_rating_30d` signal in the
//! `provider_reputation` view. The `payee` is denormalized from the
//! pre-state Channel object so the view's per-provider aggregation
//! avoids an extra join.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_channel_ratings;
use indexer_alt_schema::soma::StoredChannelRating;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::object::Object;
use types::transaction::TransactionKind;

pub struct SomaChannelRatings;

#[async_trait]
impl Processor for SomaChannelRatings {
    const NAME: &'static str = "soma_channel_ratings";

    type Value = StoredChannelRating;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, object_set, .. } = checkpoint.as_ref();
        let cp = summary.sequence_number as i64;
        let timestamp_ms = summary.timestamp_ms as i64;

        let mut out = Vec::new();
        for tx in transactions {
            let TransactionKind::RateChannel(args) = tx.transaction.kind() else {
                continue;
            };

            // Read the channel from input objects (RateChannel is a
            // read-only op so the input object IS the post object).
            let pre: Vec<&Object> = tx.input_objects(object_set).collect();
            let Some((_id, chan)) = pre
                .iter()
                .find_map(|o| {
                    if o.id() == args.channel_id {
                        o.as_channel().map(|c| (o.id(), c))
                    } else {
                        None
                    }
                })
            else {
                continue;
            };

            out.push(StoredChannelRating {
                channel_id: args.channel_id.to_vec(),
                payer: chan.payer().to_vec(),
                payee: chan.payee().to_vec(),
                negative: args.negative,
                rated_at_cp: cp,
                rated_at_ms: timestamp_ms,
            });
        }
        Ok(out)
    }
}

#[async_trait]
impl Handler for SomaChannelRatings {
    const MIN_EAGER_ROWS: usize = 10;
    const MAX_PENDING_ROWS: usize = 1_000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        // Latest-wins: UPSERT on channel_id replaces the prior
        // rating (and rated_at_*) when a payer re-rates the same
        // channel after experience changes.
        Ok(diesel::insert_into(soma_channel_ratings::table)
            .values(values)
            .on_conflict(soma_channel_ratings::channel_id)
            .do_update()
            .set((
                soma_channel_ratings::negative.eq(diesel::dsl::sql::<diesel::sql_types::Bool>(
                    "EXCLUDED.negative",
                )),
                soma_channel_ratings::rated_at_cp.eq(diesel::dsl::sql::<diesel::sql_types::Int8>(
                    "EXCLUDED.rated_at_cp",
                )),
                soma_channel_ratings::rated_at_ms.eq(diesel::dsl::sql::<diesel::sql_types::Int8>(
                    "EXCLUDED.rated_at_ms",
                )),
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
        Ok(0)
    }
}
