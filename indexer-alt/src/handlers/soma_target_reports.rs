// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_target_reports;
use indexer_alt_schema::soma::StoredTargetReport;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;

pub struct SomaTargetReports;

#[async_trait]
impl Processor for SomaTargetReports {
    const NAME: &'static str = "soma_target_reports";

    type Value = StoredTargetReport;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let cp_sequence_number = checkpoint.summary.sequence_number as i64;
        let mut values = vec![];

        for tx in &checkpoint.transactions {
            for obj in tx.output_objects(&checkpoint.object_set) {
                if let Some(target) = obj.as_target() {
                    for reporter in &target.submission_reports {
                        values.push(StoredTargetReport {
                            target_id: obj.id().to_vec(),
                            cp_sequence_number,
                            reporter: reporter.to_vec(),
                        });
                    }
                }
            }
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaTargetReports {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_target_reports::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
