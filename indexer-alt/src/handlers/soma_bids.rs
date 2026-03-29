// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_bids;
use indexer_alt_schema::soma::StoredBid;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::bid::Bid;
use types::full_checkpoint_content::Checkpoint;
use types::object::ObjectType;

pub struct SomaBids;

#[async_trait]
impl Processor for SomaBids {
    const NAME: &'static str = "soma_bids";

    type Value = StoredBid;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { summary, transactions, object_set, .. } = checkpoint.as_ref();
        let cp_sequence_number = summary.sequence_number as i64;

        let mut values = Vec::new();
        for tx in transactions.iter() {
            for obj in tx.output_objects(object_set) {
                if let Some(bid) = obj.deserialize_contents::<Bid>(ObjectType::Bid) {
                    values.push(StoredBid {
                        bid_id: bid.id.into_bytes().to_vec(),
                        cp_sequence_number,
                        ask_id: bid.ask_id.into_bytes().to_vec(),
                        seller: bid.seller.to_vec(),
                        price: bid.price as i64,
                        response_digest: bid.response_digest.into_inner().to_vec(),
                        created_at_ms: bid.created_at_ms as i64,
                        status: bid.status.to_string(),
                    });
                }
            }
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaBids {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_bids::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
