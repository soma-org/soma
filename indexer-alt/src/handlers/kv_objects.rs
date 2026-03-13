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
use indexer_alt_schema::objects::StoredObject;
use indexer_alt_schema::schema::kv_objects;

pub struct KvObjects;

#[async_trait]
impl Processor for KvObjects {
    const NAME: &'static str = "kv_objects";
    type Value = StoredObject;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let cp_sequence_number = checkpoint.summary.sequence_number as i64;

        let deleted_objects = checkpoint
            .eventually_removed_object_refs_post_version()
            .into_iter()
            .map(|(id, version, _)| {
                Ok(StoredObject {
                    object_id: id.to_vec(),
                    object_version: version.value() as i64,
                    serialized_object: None,
                    cp_sequence_number,
                })
            });

        let created_objects = checkpoint.transactions.iter().flat_map(|txn| {
            txn.output_objects(&checkpoint.object_set).map(|o| {
                let id = o.id();
                let version = o.version();
                Ok(StoredObject {
                    object_id: id.to_vec(),
                    object_version: version.value() as i64,
                    serialized_object: Some(bcs::to_bytes(o).with_context(|| {
                        format!("Serializing object {id} version {}", version.value())
                    })?),
                    cp_sequence_number,
                })
            })
        });

        deleted_objects
            .chain(created_objects)
            .collect::<Result<Vec<_>, _>>()
    }
}

#[async_trait]
impl Handler for KvObjects {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(kv_objects::table)
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
        Ok(diesel::delete(
            kv_objects::table
                .filter(kv_objects::cp_sequence_number.ge(from as i64))
                .filter(kv_objects::cp_sequence_number.lt(to_exclusive as i64)),
        )
        .execute(conn)
        .await?)
    }
}
