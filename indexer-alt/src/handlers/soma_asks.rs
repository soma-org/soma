// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_asks;
use indexer_alt_schema::soma::StoredAsk;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::ask::{Ask, AskStatus};
use types::full_checkpoint_content::Checkpoint;
use types::object::ObjectType;

pub struct SomaAsks;

fn status_label(status: &AskStatus) -> &'static str {
    match status {
        AskStatus::Open => "open",
        AskStatus::Filled => "filled",
        AskStatus::Cancelled => "cancelled",
        AskStatus::Expired => "expired",
    }
}

#[async_trait]
impl Processor for SomaAsks {
    const NAME: &'static str = "soma_asks";

    type Value = StoredAsk;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { summary, transactions, object_set, .. } = checkpoint.as_ref();
        let cp_sequence_number = summary.sequence_number as i64;

        let mut values = Vec::new();
        for tx in transactions.iter() {
            for obj in tx.output_objects(object_set) {
                if let Some(ask) = obj.deserialize_contents::<Ask>(ObjectType::Ask) {
                    values.push(StoredAsk {
                        ask_id: ask.id.into_bytes().to_vec(),
                        cp_sequence_number,
                        buyer: ask.buyer.to_vec(),
                        task_digest: ask.task_digest.into_inner().to_vec(),
                        max_price_per_bid: ask.max_price_per_bid as i64,
                        num_bids_wanted: ask.num_bids_wanted as i32,
                        timeout_ms: ask.timeout_ms as i64,
                        created_at_ms: ask.created_at_ms as i64,
                        status: status_label(&ask.status).to_string(),
                        accepted_bid_count: ask.accepted_bid_count as i32,
                    });
                }
            }
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaAsks {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_asks::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
