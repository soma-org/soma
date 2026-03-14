// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_staked_soma;
use indexer_alt_schema::soma::StoredStakedSoma;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::effects::TransactionEffectsAPI;
use types::full_checkpoint_content::Checkpoint;
use types::object::{ObjectType, Owner};

use crate::handlers::checkpoint_input_objects;

pub struct SomaStakedSoma;

#[async_trait]
impl Processor for SomaStakedSoma {
    const NAME: &'static str = "soma_staked_soma";

    type Value = StoredStakedSoma;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, object_set, .. } = checkpoint.as_ref();

        let cp_sequence_number = summary.sequence_number as i64;
        let input_objects = checkpoint_input_objects(checkpoint)?;

        let mut values = Vec::new();

        for tx in transactions {
            // Tombstone entries for deleted/wrapped StakedSoma objects
            for change in tx.effects.object_changes() {
                if change.output_version.is_some() {
                    continue;
                }
                // Check if the deleted object was a StakedSoma in the inputs
                if let Some(input_obj) = input_objects.get(&change.id) {
                    if *input_obj.data.object_type() == ObjectType::StakedSoma {
                        values.push(StoredStakedSoma {
                            staked_soma_id: change.id.to_vec(),
                            cp_sequence_number,
                            owner: None,
                            pool_id: None,
                            stake_activation_epoch: None,
                            principal: None,
                        });
                    }
                }
            }

            // Output StakedSoma objects
            for obj in tx.output_objects(object_set) {
                if let Some(staked) = obj.as_staked_soma() {
                    let owner = match obj.owner() {
                        Owner::AddressOwner(addr) => addr.to_vec(),
                        _ => continue,
                    };

                    values.push(StoredStakedSoma {
                        staked_soma_id: obj.id().to_vec(),
                        cp_sequence_number,
                        owner: Some(owner),
                        pool_id: Some(staked.pool_id.to_vec()),
                        stake_activation_epoch: Some(staked.stake_activation_epoch as i64),
                        principal: Some(staked.principal as i64),
                    });
                }
            }
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaStakedSoma {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_staked_soma::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
