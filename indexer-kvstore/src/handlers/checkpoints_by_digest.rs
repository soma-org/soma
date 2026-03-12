// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use indexer_framework::pipeline::Processor;
use types::full_checkpoint_content::Checkpoint;

use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::handlers::BigTableProcessor;
use crate::tables;

pub struct CheckpointsByDigestPipeline;

#[async_trait::async_trait]
impl Processor for CheckpointsByDigestPipeline {
    const NAME: &'static str = "kvstore_checkpoints_by_digest";
    type Value = Entry;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        let summary = &checkpoint.summary;
        let timestamp_ms = summary.timestamp_ms;

        let entry = tables::make_entry(
            tables::checkpoints_by_digest::encode_key(summary.digest()),
            tables::checkpoints_by_digest::encode(summary.sequence_number),
            Some(timestamp_ms),
        );

        Ok(vec![entry])
    }
}

impl BigTableProcessor for CheckpointsByDigestPipeline {
    const TABLE: &'static str = tables::checkpoints_by_digest::NAME;
}
