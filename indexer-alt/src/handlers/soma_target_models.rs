// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_target_models;
use indexer_alt_schema::soma::StoredTargetModel;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;

pub struct SomaTargetModels;

#[async_trait]
impl Processor for SomaTargetModels {
    const NAME: &'static str = "soma_target_models";

    type Value = StoredTargetModel;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let cp_sequence_number = checkpoint.summary.sequence_number as i64;
        let mut values = vec![];

        for tx in &checkpoint.transactions {
            for obj in tx.output_objects(&checkpoint.object_set) {
                if let Some(target) = obj.as_target() {
                    let target_id = obj.id().to_vec();
                    for model_id in &target.model_ids {
                        values.push(StoredTargetModel {
                            target_id: target_id.clone(),
                            cp_sequence_number,
                            model_id: model_id.to_vec(),
                        });
                    }
                }
            }
        }
        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaTargetModels {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_target_models::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
