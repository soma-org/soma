// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use indexer_framework::pipeline::Processor;
use types::full_checkpoint_content::Checkpoint;
use types::object::ObjectType;

use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::handlers::BigTableProcessor;
use crate::tables;

/// KV pipeline that writes target objects to BigTable.
pub struct SomaTargetsPipeline;

#[async_trait::async_trait]
impl Processor for SomaTargetsPipeline {
    const NAME: &'static str = "kvstore_soma_targets";
    type Value = Entry;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        let timestamp_ms = checkpoint.summary.timestamp_ms;
        let mut entries = vec![];

        for tx in &checkpoint.transactions {
            for obj in tx.output_objects(&checkpoint.object_set) {
                if !matches!(obj.data.object_type(), ObjectType::Target) {
                    continue;
                }
                if let Some(target) = obj.as_target() {
                    let state_bcs = bcs::to_bytes(&target)?;
                    let entry = tables::make_entry(
                        tables::targets::encode_key(&obj.id().to_vec()),
                        tables::targets::encode(&state_bcs),
                        Some(timestamp_ms),
                    );
                    entries.push(entry);
                }
            }
        }

        Ok(entries)
    }
}

impl BigTableProcessor for SomaTargetsPipeline {
    const TABLE: &'static str = tables::targets::NAME;
}
