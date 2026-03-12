// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context as _;
use indexer_framework::pipeline::Processor;
use types::full_checkpoint_content::Checkpoint;
use types::system_state::{SystemStateTrait as _, get_system_state};
use types::transaction::TransactionKind;

use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::handlers::BigTableProcessor;
use crate::tables;

/// KV pipeline that writes model registry snapshots at epoch boundaries.
pub struct SomaModelsPipeline;

#[async_trait::async_trait]
impl Processor for SomaModelsPipeline {
    const NAME: &'static str = "kvstore_soma_models";
    type Value = Entry;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        let summary = &checkpoint.summary;

        // Only process epoch-boundary checkpoints
        if summary.sequence_number != 0 && summary.end_of_epoch_data.is_none() {
            return Ok(vec![]);
        }

        let tx = if summary.sequence_number == 0 {
            &checkpoint.transactions[0]
        } else {
            checkpoint
                .transactions
                .iter()
                .find(|tx| matches!(tx.transaction.kind(), TransactionKind::ChangeEpoch(_)))
                .context("No ChangeEpoch tx in end-of-epoch checkpoint")?
        };

        let output_objects: Vec<_> = tx.output_objects(&checkpoint.object_set).cloned().collect();
        let system_state = get_system_state(&output_objects.as_slice())?;
        let registry = system_state.model_registry();
        let epoch = system_state.epoch();
        let timestamp_ms = summary.timestamp_ms;

        let mut entries = vec![];

        for (model_id, model) in &registry.active_models {
            let state_bcs = bcs::to_bytes(model)?;
            let entry = tables::make_entry(
                tables::models::encode_key(&model_id.to_vec(), epoch),
                tables::models::encode(&state_bcs),
                Some(timestamp_ms),
            );
            entries.push(entry);
        }
        for (model_id, model) in &registry.pending_models {
            let state_bcs = bcs::to_bytes(model)?;
            let entry = tables::make_entry(
                tables::models::encode_key(&model_id.to_vec(), epoch),
                tables::models::encode(&state_bcs),
                Some(timestamp_ms),
            );
            entries.push(entry);
        }
        for (model_id, model) in &registry.inactive_models {
            let state_bcs = bcs::to_bytes(model)?;
            let entry = tables::make_entry(
                tables::models::encode_key(&model_id.to_vec(), epoch),
                tables::models::encode(&state_bcs),
                Some(timestamp_ms),
            );
            entries.push(entry);
        }

        Ok(entries)
    }
}

impl BigTableProcessor for SomaModelsPipeline {
    const TABLE: &'static str = tables::models::NAME;
    const MIN_EAGER_ROWS: usize = 1;
}
