// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::epochs::StoredEpochStart;
use indexer_alt_schema::schema::kv_epoch_starts;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::system_state::{SystemStateTrait as _, get_system_state};
use types::transaction::TransactionKind;

use crate::handlers::cp_sequence_numbers::epoch_interval;

pub struct KvEpochStarts;

#[async_trait]
impl Processor for KvEpochStarts {
    const NAME: &'static str = "kv_epoch_starts";

    type Value = StoredEpochStart;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { summary, transactions, object_set, .. } = checkpoint.as_ref();

        // Epoch start info comes from the last checkpoint of the *previous* epoch, which
        // contains the ChangeEpoch transaction whose output objects include the new system state.
        // We also handle checkpoint 0 (genesis) which bootstraps epoch 0.

        if summary.sequence_number == 0 {
            // Genesis checkpoint: extract epoch 0 start info from genesis transaction outputs
            if transactions.is_empty() {
                anyhow::bail!("Genesis checkpoint has no transactions");
            }
            let tx = &transactions[0];

            // Collect output objects for system state lookup
            let output_objects: Vec<_> = tx.output_objects(object_set).cloned().collect();
            let system_state = get_system_state(&output_objects.as_slice())
                .context("Failed to find system state object in genesis transaction outputs")?;

            return Ok(vec![StoredEpochStart {
                epoch: system_state.epoch() as i64,
                protocol_version: system_state.protocol_version() as i64,
                cp_lo: 0,
                start_timestamp_ms: system_state.epoch_start_timestamp_ms() as i64,
                reference_gas_price: 0, // Soma uses fee parameters, not reference gas price
                system_state: bcs::to_bytes(&system_state)
                    .context("Failed to serialize SystemState")?,
            }]);
        }

        // For non-genesis checkpoints, only process end-of-epoch checkpoints
        if summary.end_of_epoch_data.is_none() {
            return Ok(vec![]);
        }

        // Find the ChangeEpoch transaction
        let Some(tx) = transactions
            .iter()
            .find(|tx| matches!(tx.transaction.kind(), TransactionKind::ChangeEpoch(_)))
        else {
            bail!(
                "Failed to get ChangeEpoch transaction in checkpoint {} with EndOfEpochData",
                summary.sequence_number,
            );
        };

        // Get output objects and extract system state
        let output_objects: Vec<_> = tx.output_objects(object_set).cloned().collect();
        let system_state = get_system_state(&output_objects.as_slice()).with_context(|| {
            format!(
                "Failed to find system state object in ChangeEpoch transaction outputs at checkpoint {}",
                summary.sequence_number,
            )
        })?;

        Ok(vec![StoredEpochStart {
            epoch: system_state.epoch() as i64,
            protocol_version: system_state.protocol_version() as i64,
            cp_lo: summary.sequence_number as i64 + 1,
            start_timestamp_ms: system_state.epoch_start_timestamp_ms() as i64,
            reference_gas_price: 0, // Soma uses fee parameters, not reference gas price
            system_state: bcs::to_bytes(&system_state)
                .context("Failed to serialize SystemState")?,
        }])
    }
}

#[async_trait]
impl Handler for KvEpochStarts {
    const MIN_EAGER_ROWS: usize = 1;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(kv_epoch_starts::table)
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
        let Range { start: from_epoch, end: to_epoch } =
            epoch_interval(conn, from..to_exclusive).await?;
        if from_epoch < to_epoch {
            let filter = kv_epoch_starts::table
                .filter(kv_epoch_starts::epoch.between(from_epoch as i64, to_epoch as i64 - 1));
            Ok(diesel::delete(filter).execute(conn).await?)
        } else {
            Ok(0)
        }
    }
}
