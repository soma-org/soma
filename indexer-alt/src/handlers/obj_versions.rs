// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::objects::StoredObjVersion;
use indexer_alt_schema::schema::obj_versions;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::effects::TransactionEffectsAPI;
use types::full_checkpoint_content::Checkpoint;

pub struct ObjVersions;

#[async_trait]
impl Processor for ObjVersions {
    const NAME: &'static str = "obj_versions";

    type Value = StoredObjVersion;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, .. } = checkpoint.as_ref();

        let cp_sequence_number = summary.sequence_number as i64;

        let mut values = Vec::new();

        for tx in transactions {
            for change in tx.effects.object_changes() {
                let object_id = change.id.to_vec();

                // Use the output version if available, otherwise the lamport version
                let (object_version, object_digest) = if let Some(version) = change.output_version {
                    (version.value() as i64, change.output_digest.map(|d| d.inner().to_vec()))
                } else {
                    // Object was deleted or wrapped -- use the lamport version from effects
                    (tx.effects.version().value() as i64, None)
                };

                values.push(StoredObjVersion {
                    object_id,
                    object_version,
                    object_digest,
                    cp_sequence_number,
                });
            }
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for ObjVersions {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(obj_versions::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
