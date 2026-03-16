// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_validators;
use indexer_alt_schema::soma::StoredValidator;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::system_state::{SystemStateTrait as _, get_system_state};
use types::transaction::TransactionKind;

pub struct SomaValidators;

#[async_trait]
impl Processor for SomaValidators {
    const NAME: &'static str = "soma_validators";

    type Value = StoredValidator;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { summary, transactions, object_set, .. } = checkpoint.as_ref();

        // Validators are extracted at epoch boundaries from the system state,
        // same as kv_epoch_starts. We handle genesis (checkpoint 0) and
        // end-of-epoch checkpoints with ChangeEpoch transactions.

        if summary.sequence_number == 0 {
            // Genesis checkpoint
            if transactions.is_empty() {
                bail!("Genesis checkpoint has no transactions");
            }
            let tx = &transactions[0];
            let output_objects: Vec<_> = tx.output_objects(object_set).cloned().collect();
            let system_state = get_system_state(&output_objects.as_slice())?;
            return Ok(extract_validators(&system_state));
        }

        // Only process end-of-epoch checkpoints
        if summary.end_of_epoch_data.is_none() {
            return Ok(vec![]);
        }

        // Find the ChangeEpoch transaction
        let Some(tx) = transactions
            .iter()
            .find(|tx| matches!(tx.transaction.kind(), TransactionKind::ChangeEpoch(_)))
        else {
            bail!(
                "No ChangeEpoch transaction in checkpoint {} with EndOfEpochData",
                summary.sequence_number,
            );
        };

        let output_objects: Vec<_> = tx.output_objects(object_set).cloned().collect();
        let system_state = get_system_state(&output_objects.as_slice())?;
        Ok(extract_validators(&system_state))
    }
}

fn extract_validators(system_state: &types::system_state::SystemState) -> Vec<StoredValidator> {
    let epoch = system_state.epoch() as i64;
    let validator_set = system_state.validators();

    validator_set
        .validators
        .iter()
        .map(|v| {
            StoredValidator {
                address: v.metadata.soma_address.to_vec(),
                epoch,
                voting_power: v.voting_power as i64,
                commission_rate: v.commission_rate as i64,
                next_epoch_commission_rate: v.next_epoch_commission_rate as i64,
                staking_pool_id: v.staking_pool.id.to_vec(),
                stake: v.staking_pool.soma_balance as i64,
                pending_stake: v.staking_pool.pending_stake as i64,
                name: None, // ValidatorMetadata doesn't have a name field
                network_address: Some(v.metadata.net_address.to_string()),
                proxy_address: Some(v.metadata.proxy_address.to_string()),
                protocol_pubkey: bcs::to_bytes(&v.metadata.protocol_pubkey).ok(),
            }
        })
        .collect()
}

#[async_trait]
impl Handler for SomaValidators {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_validators::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
