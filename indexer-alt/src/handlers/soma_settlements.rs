// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_settlements;
use indexer_alt_schema::soma::StoredSettlement;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::object::ObjectType;
use types::settlement::Settlement;

pub struct SomaSettlements;

#[async_trait]
impl Processor for SomaSettlements {
    const NAME: &'static str = "soma_settlements";

    type Value = StoredSettlement;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { summary, transactions, object_set, .. } = checkpoint.as_ref();
        let cp_sequence_number = summary.sequence_number as i64;

        let mut values = Vec::new();
        for tx in transactions.iter() {
            for obj in tx.output_objects(object_set) {
                if let Some(s) = obj.deserialize_contents::<Settlement>(ObjectType::Settlement) {
                    values.push(StoredSettlement {
                        settlement_id: s.id.into_bytes().to_vec(),
                        cp_sequence_number,
                        ask_id: s.ask_id.into_bytes().to_vec(),
                        bid_id: s.bid_id.into_bytes().to_vec(),
                        buyer: s.buyer.to_vec(),
                        seller: s.seller.to_vec(),
                        amount: s.amount as i64,
                        task_digest: s.task_digest.into_inner().to_vec(),
                        response_digest: s.response_digest.into_inner().to_vec(),
                        settled_at_ms: s.settled_at_ms as i64,
                        seller_rating: s.seller_rating.to_string(),
                        rating_deadline_ms: s.rating_deadline_ms as i64,
                    });
                }
            }
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaSettlements {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_settlements::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
