// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_vaults;
use indexer_alt_schema::soma::StoredVault;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::object::ObjectType;
use types::vault::SellerVault;

pub struct SomaVaults;

#[async_trait]
impl Processor for SomaVaults {
    const NAME: &'static str = "soma_vaults";

    type Value = StoredVault;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { summary, transactions, object_set, .. } = checkpoint.as_ref();
        let cp_sequence_number = summary.sequence_number as i64;

        let mut values = Vec::new();
        for tx in transactions.iter() {
            for obj in tx.output_objects(object_set) {
                if let Some(v) = obj.deserialize_contents::<SellerVault>(ObjectType::SellerVault) {
                    values.push(StoredVault {
                        vault_id: v.id.into_bytes().to_vec(),
                        cp_sequence_number,
                        owner: v.owner.to_vec(),
                        balance: v.balance as i64,
                    });
                }
            }
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaVaults {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_vaults::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
