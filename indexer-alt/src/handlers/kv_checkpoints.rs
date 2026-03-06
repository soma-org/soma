// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use indexer_alt_schema::checkpoints::StoredCheckpoint;
use indexer_alt_schema::schema::kv_checkpoints;

pub struct KvCheckpoints;

#[async_trait]
impl Processor for KvCheckpoints {
    const NAME: &'static str = "kv_checkpoints";

    type Value = StoredCheckpoint;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint {
            summary,
            contents,
            ..
        } = checkpoint.as_ref();

        let sequence_number = summary.sequence_number as i64;

        let checkpoint_contents =
            bcs::to_bytes(contents).context("Serializing CheckpointContents")?;

        let checkpoint_summary =
            bcs::to_bytes(summary.data()).context("Serializing CheckpointSummary")?;

        let validator_signatures =
            bcs::to_bytes(summary.auth_sig()).context("Serializing validator signatures")?;

        Ok(vec![StoredCheckpoint {
            sequence_number,
            checkpoint_contents,
            checkpoint_summary,
            validator_signatures,
        }])
    }
}

#[async_trait]
impl Handler for KvCheckpoints {
    const MIN_EAGER_ROWS: usize = 1;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(kv_checkpoints::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }

    async fn prune<'a>(
        &self,
        from: u64,
        to_exclusive: u64,
        conn: &mut Connection<'a>,
    ) -> Result<usize> {
        let filter = kv_checkpoints::table.filter(
            kv_checkpoints::sequence_number.between(from as i64, to_exclusive as i64 - 1),
        );

        Ok(diesel::delete(filter).execute(conn).await?)
    }
}
