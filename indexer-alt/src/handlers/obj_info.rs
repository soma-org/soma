// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::objects::{StoredObjInfo, StoredOwnerKind};
use indexer_alt_schema::schema::obj_info;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::effects::TransactionEffectsAPI;
use types::full_checkpoint_content::Checkpoint;
use types::object::Owner;

use crate::handlers::checkpoint_input_objects;

pub struct ObjInfo;

#[async_trait]
impl Processor for ObjInfo {
    const NAME: &'static str = "obj_info";

    type Value = StoredObjInfo;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, object_set, .. } = checkpoint.as_ref();

        let cp_sequence_number = summary.sequence_number as i64;
        let input_objects = checkpoint_input_objects(checkpoint)?;

        let mut values = Vec::new();

        for tx in transactions {
            // Emit tombstone entries for deleted/wrapped objects (no output version)
            for change in tx.effects.object_changes() {
                if change.output_version.is_some() {
                    continue;
                }
                values.push(StoredObjInfo {
                    object_id: change.id.to_vec(),
                    cp_sequence_number,
                    owner_kind: None,
                    owner_id: None,
                    package: None,
                    module: None,
                    name: None,
                    instantiation: None,
                });
            }

            // Emit entries for output objects (created or mutated)
            for obj in tx.output_objects(object_set) {
                let (owner_kind, owner_id) = owner_info(obj.owner());

                // Check if ownership actually changed compared to the input
                let ownership_changed = match input_objects.get(&obj.id()) {
                    Some(old) => {
                        let (old_kind, old_id) = owner_info(old.owner());
                        old_kind != owner_kind || old_id != owner_id
                    }
                    None => true, // New object
                };

                if !ownership_changed {
                    continue;
                }

                // Soma has simple object types (Coin, StakedSoma, Target, SystemState)
                // rather than Move struct tags (package/module/name/instantiation).
                let type_name = obj.type_().to_string();
                values.push(StoredObjInfo {
                    object_id: obj.id().to_vec(),
                    cp_sequence_number,
                    owner_kind: Some(owner_kind),
                    owner_id: Some(owner_id),
                    package: None,
                    module: Some(type_name),
                    name: None,
                    instantiation: None,
                });
            }
        }

        Ok(values)
    }
}

/// Convert a Soma Owner to the stored owner kind and ID bytes.
fn owner_info(owner: &Owner) -> (StoredOwnerKind, Vec<u8>) {
    match owner {
        Owner::AddressOwner(address) => (StoredOwnerKind::Address, address.to_vec()),
        Owner::Shared { .. } => (StoredOwnerKind::Shared, vec![]),
        Owner::Immutable => (StoredOwnerKind::Immutable, vec![]),
        // Stage 14a: accumulator objects have no externally-meaningful
        // owner address. The accumulator kind (Balance vs Delegation)
        // is recoverable from the object's ObjectType when needed.
        Owner::Accumulator { .. } => (StoredOwnerKind::Accumulator, vec![]),
    }
}

#[async_trait]
impl Handler for ObjInfo {
    const MIN_EAGER_ROWS: usize = 100;
    const MAX_PENDING_ROWS: usize = 10000;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(obj_info::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
