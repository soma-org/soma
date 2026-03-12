// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::model::ModelId;
use types::model::ModelV1;
use types::system_state::{SystemStateTrait as _, get_system_state};
use types::transaction::TransactionKind;
use indexer_alt_schema::schema::soma_models;
use indexer_alt_schema::soma::StoredModel;

pub struct SomaModels;

fn stored_model(model_id: ModelId, model: &ModelV1, epoch: i64, status: &str) -> Result<StoredModel> {
    Ok(StoredModel {
        model_id: model_id.to_vec(),
        epoch,
        status: status.to_string(),
        owner: model.owner.to_vec(),
        architecture_version: model.architecture_version as i64,
        commit_epoch: model.commit_epoch as i64,
        stake: model.stake() as i64,
        commission_rate: model.commission_rate as i64,
        has_embedding: model.embedding.is_some(),
        state_bcs: bcs::to_bytes(model).context("Serializing ModelV1")?,
    })
}

#[async_trait]
impl Processor for SomaModels {
    const NAME: &'static str = "soma_models";

    type Value = StoredModel;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint {
            summary,
            transactions,
            object_set,
            ..
        } = checkpoint.as_ref();

        // Only process epoch-boundary checkpoints (genesis or end-of-epoch)
        if summary.sequence_number != 0 && summary.end_of_epoch_data.is_none() {
            return Ok(vec![]);
        }

        // Find the relevant transaction containing the system state
        let tx = if summary.sequence_number == 0 {
            // Genesis checkpoint
            if transactions.is_empty() {
                bail!("Genesis checkpoint has no transactions");
            }
            &transactions[0]
        } else {
            // End-of-epoch: find the ChangeEpoch transaction
            transactions.iter()
                .find(|tx| matches!(tx.transaction.kind(), TransactionKind::ChangeEpoch(_)))
                .context("No ChangeEpoch tx in end-of-epoch checkpoint")?
        };

        let output_objects: Vec<_> = tx.output_objects(object_set).cloned().collect();
        let system_state = get_system_state(&output_objects.as_slice())
            .context("Failed to find system state in epoch boundary transaction")?;
        let registry = system_state.model_registry();
        let epoch = system_state.epoch() as i64;

        let mut values = vec![];
        for (model_id, model) in &registry.active_models {
            values.push(stored_model(*model_id, model, epoch, "active")?);
        }
        for (model_id, model) in &registry.pending_models {
            values.push(stored_model(*model_id, model, epoch, "pending")?);
        }
        for (model_id, model) in &registry.inactive_models {
            values.push(stored_model(*model_id, model, epoch, "inactive")?);
        }
        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaModels {
    const MIN_EAGER_ROWS: usize = 1;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_models::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
