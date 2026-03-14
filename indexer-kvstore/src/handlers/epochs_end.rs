// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context as _;
use anyhow::bail;
use indexer_framework::pipeline::Processor;
use types::full_checkpoint_content::Checkpoint;
use types::transaction::TransactionKind;

use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::handlers::BigTableProcessor;
use crate::tables;

pub struct EpochEndPipeline;

#[async_trait::async_trait]
impl Processor for EpochEndPipeline {
    const NAME: &'static str = "kvstore_epochs_end";
    type Value = Entry;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        let summary = &checkpoint.summary;

        let Some(end_of_epoch) = summary.end_of_epoch_data.as_ref() else {
            return Ok(vec![]);
        };

        let Some(_transaction) = checkpoint
            .transactions
            .iter()
            .find(|tx| matches!(tx.transaction.kind(), TransactionKind::ChangeEpoch(_)))
        else {
            bail!(
                "No ChangeEpoch tx in checkpoint {} with EndOfEpochData",
                summary.sequence_number,
            );
        };

        let epoch_id = summary.epoch;
        let end_timestamp_ms = summary.timestamp_ms;
        let end_checkpoint = summary.sequence_number;
        let cp_hi = summary.sequence_number + 1;
        let tx_hi = summary.network_total_transactions;

        let epoch_commitments = bcs::to_bytes(&end_of_epoch.epoch_commitments)
            .context("Failed to serialize EpochCommitment-s")?;

        // Soma has no SystemEpochInfoEvent, so staking/storage fields are omitted.
        let entry = tables::make_entry(
            tables::epochs::encode_key(epoch_id),
            tables::epochs::encode_end(
                end_timestamp_ms,
                end_checkpoint,
                cp_hi,
                tx_hi,
                false, // safe_mode
                &epoch_commitments,
            ),
            Some(end_timestamp_ms),
        );

        Ok(vec![entry])
    }
}

impl BigTableProcessor for EpochEndPipeline {
    const TABLE: &'static str = tables::epochs::NAME;
    const MIN_EAGER_ROWS: usize = 1;
}
