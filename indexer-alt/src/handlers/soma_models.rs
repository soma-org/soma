// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_models;
use indexer_alt_schema::soma::StoredModel;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::metadata::{ManifestAPI, MetadataAPI};
use types::model::{Model, ModelId, ModelStateV1};
use types::system_state::{SystemStateTrait as _, get_system_state};
use types::transaction::TransactionKind;

pub struct SomaModels;

fn stored_model(model_id: ModelId, model: &Model, epoch: i64) -> Result<StoredModel> {
    let status = match model {
        Model::V1(ModelStateV1::Created(_)) => "created",
        Model::V1(ModelStateV1::Pending(_)) => "pending",
        Model::V1(ModelStateV1::Active(_)) => "active",
        Model::V1(ModelStateV1::Inactive(_)) => "inactive",
    };

    // commit_epoch: meaningful for Pending; for Created use create_epoch; otherwise 0
    let commit_epoch = model
        .commit_epoch()
        .or(model.create_epoch())
        .unwrap_or(0) as i64;

    // StakingPool fields
    let pool = model.staking_pool();
    let exchange_rates_json =
        serde_json::to_string(&pool.exchange_rates).unwrap_or_else(|_| "{}".to_string());

    // Manifest fields (available on Pending/Active/Inactive)
    let (manifest_url, manifest_checksum, manifest_size) = match model.manifest() {
        Some(m) => {
            let url = m.url().to_string();
            let meta = m.metadata();
            let checksum = meta.checksum().as_ref().to_vec();
            let size = meta.size() as i64;
            (Some(url), Some(checksum), Some(size))
        }
        None => (None, None, None),
    };

    // Weights commitment (available on Pending/Active/Inactive)
    let weights_commitment = match model {
        Model::V1(ModelStateV1::Pending(m)) => {
            Some(AsRef::<[u8]>::as_ref(&m.weights_commitment).to_vec())
        }
        Model::V1(ModelStateV1::Active(m)) => {
            Some(AsRef::<[u8]>::as_ref(&m.weights_commitment).to_vec())
        }
        Model::V1(ModelStateV1::Inactive(m)) => {
            Some(AsRef::<[u8]>::as_ref(&m.weights_commitment).to_vec())
        }
        _ => None,
    };

    // Pending model update fields (only on Active models with a pending update)
    let (pu_url, pu_checksum, pu_size, pu_wc, pu_epoch) =
        match model.as_active().and_then(|a| a.pending_update.as_ref()) {
            Some(pu) => {
                let m = &pu.manifest;
                let url = m.url().to_string();
                let meta = m.metadata();
                (
                    Some(url),
                    Some(meta.checksum().as_ref().to_vec()),
                    Some(meta.size() as i64),
                    Some(AsRef::<[u8]>::as_ref(&pu.weights_commitment).to_vec()),
                    Some(pu.commit_epoch as i64),
                )
            }
            None => (None, None, None, None, None),
        };

    Ok(StoredModel {
        model_id: model_id.to_vec(),
        epoch,
        status: status.to_string(),
        owner: model.owner().to_vec(),
        architecture_version: model.architecture_version() as i64,
        commit_epoch,
        stake: model.stake() as i64,
        commission_rate: model.commission_rate() as i64,
        next_epoch_commission_rate: model.next_epoch_commission_rate() as i64,
        staking_pool_id: pool.id.to_vec(),
        activation_epoch: pool.activation_epoch.map(|e| e as i64),
        deactivation_epoch: pool.deactivation_epoch.map(|e| e as i64),
        rewards_pool: pool.rewards_pool as i64,
        pool_token_balance: pool.pool_token_balance as i64,
        pending_stake: pool.pending_stake as i64,
        pending_total_soma_withdraw: pool.pending_total_soma_withdraw as i64,
        pending_pool_token_withdraw: pool.pending_pool_token_withdraw as i64,
        exchange_rates_json,
        manifest_url,
        manifest_checksum,
        manifest_size,
        weights_commitment,
        has_pending_update: model.has_pending_update(),
        pending_manifest_url: pu_url,
        pending_manifest_checksum: pu_checksum,
        pending_manifest_size: pu_size,
        pending_weights_commitment: pu_wc,
        pending_commit_epoch: pu_epoch,
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
        for (model_id, model) in &registry.models {
            values.push(stored_model(*model_id, model, epoch)?);
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
