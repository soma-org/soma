// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context as _;
use anyhow::bail;
use indexer_framework::pipeline::Processor;
use types::full_checkpoint_content::Checkpoint;
use types::system_state::{SystemStateTrait as _, get_system_state};
use types::transaction::TransactionKind;

use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::handlers::BigTableProcessor;
use crate::tables;

pub struct EpochStartPipeline;

#[async_trait::async_trait]
impl Processor for EpochStartPipeline {
    const NAME: &'static str = "kvstore_epochs_start";
    type Value = Entry;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        let Checkpoint { summary, transactions, object_set, .. } = checkpoint.as_ref();

        // Only process genesis or end-of-epoch checkpoints
        if summary.sequence_number != 0 && summary.end_of_epoch_data.is_none() {
            return Ok(vec![]);
        }

        let (start_checkpoint, tx) = if summary.sequence_number == 0 {
            (0u64, &transactions[0])
        } else {
            let Some(tx) = transactions
                .iter()
                .find(|tx| matches!(tx.transaction.kind(), TransactionKind::ChangeEpoch(_)))
            else {
                bail!(
                    "No ChangeEpoch tx in checkpoint {} with EndOfEpochData",
                    summary.sequence_number,
                );
            };
            (summary.sequence_number + 1, tx)
        };

        let output_objects: Vec<_> = tx.output_objects(object_set).cloned().collect();
        let system_state =
            get_system_state(&output_objects.as_slice()).context("Failed to find system state")?;

        let epoch = system_state.epoch();
        let protocol_version = system_state.protocol_version();
        let start_timestamp_ms = system_state.epoch_start_timestamp_ms();
        let system_state_bcs =
            bcs::to_bytes(&system_state).context("Failed to serialize SystemState")?;

        let entry = tables::make_entry(
            tables::epochs::encode_key(epoch),
            tables::epochs::encode_start(
                epoch,
                protocol_version,
                start_timestamp_ms,
                start_checkpoint,
                0, // Soma has no reference gas price
                &system_state_bcs,
            ),
            Some(start_timestamp_ms),
        );

        Ok(vec![entry])
    }
}

impl BigTableProcessor for EpochStartPipeline {
    const TABLE: &'static str = tables::epochs::NAME;
    const MIN_EAGER_ROWS: usize = 1;
}
