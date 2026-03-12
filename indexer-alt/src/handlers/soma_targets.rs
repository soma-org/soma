// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::target::TargetStatus;
use indexer_alt_schema::schema::soma_targets;
use indexer_alt_schema::soma::StoredTarget;

pub struct SomaTargets;

#[async_trait]
impl Processor for SomaTargets {
    const NAME: &'static str = "soma_targets";

    type Value = StoredTarget;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let cp_sequence_number = checkpoint.summary.sequence_number as i64;
        let mut values = vec![];

        for tx in &checkpoint.transactions {
            for obj in tx.output_objects(&checkpoint.object_set) {
                if let Some(target) = obj.as_target() {
                    let status = match &target.status {
                        TargetStatus::Open => "open".to_string(),
                        TargetStatus::Filled { .. } => "filled".to_string(),
                        TargetStatus::Claimed => "claimed".to_string(),
                    };

                    values.push(StoredTarget {
                        target_id: obj.id().to_vec(),
                        cp_sequence_number,
                        epoch: target.generation_epoch as i64,
                        status,
                        submitter: target.submitter.map(|a| a.to_vec()),
                        winning_model_id: target.winning_model_id.map(|id| id.to_vec()),
                        reward_pool: target.reward_pool as i64,
                        bond_amount: target.bond_amount as i64,
                        report_count: target.submission_reports.len() as i32,
                        state_bcs: bcs::to_bytes(&target)?,
                    });
                }
            }
        }
        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaTargets {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_targets::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
