// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use indexer_framework::pipeline::Processor;
use types::full_checkpoint_content::Checkpoint;
use types::storage::ObjectKey;

use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::handlers::BigTableProcessor;
use crate::tables;

pub struct ObjectsPipeline;

#[async_trait::async_trait]
impl Processor for ObjectsPipeline {
    const NAME: &'static str = "kvstore_objects";
    type Value = Entry;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        let timestamp_ms = checkpoint.summary.timestamp_ms;
        let mut entries = vec![];

        for tx in &checkpoint.transactions {
            for object in tx.output_objects(&checkpoint.object_set) {
                let object_key = ObjectKey(object.id(), object.version());
                let entry = tables::make_entry(
                    tables::objects::encode_key(&object_key),
                    tables::objects::encode(object)?,
                    Some(timestamp_ms),
                );
                entries.push(entry);
            }
        }

        Ok(entries)
    }
}

impl BigTableProcessor for ObjectsPipeline {
    const TABLE: &'static str = tables::objects::NAME;
}
