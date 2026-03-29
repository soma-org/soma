// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_epoch_state;
use indexer_alt_schema::soma::StoredEpochState;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::system_state::{SystemState, SystemStateTrait as _, get_system_state};
use types::transaction::TransactionKind;

pub struct SomaEpochState;

#[async_trait]
impl Processor for SomaEpochState {
    const NAME: &'static str = "soma_epoch_state";

    type Value = StoredEpochState;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { summary, transactions, object_set, .. } = checkpoint.as_ref();

        // Only process epoch-boundary checkpoints
        if summary.sequence_number != 0 && summary.end_of_epoch_data.is_none() {
            return Ok(vec![]);
        }

        let tx = if summary.sequence_number == 0 {
            if transactions.is_empty() {
                bail!("Genesis checkpoint has no transactions");
            }
            &transactions[0]
        } else {
            transactions
                .iter()
                .find(|tx| matches!(tx.transaction.kind(), TransactionKind::ChangeEpoch(_)))
                .context("No ChangeEpoch tx in end-of-epoch checkpoint")?
        };

        let output_objects: Vec<_> = tx.output_objects(object_set).cloned().collect();
        let system_state = get_system_state(&output_objects.as_slice())
            .context("Failed to find system state in epoch boundary transaction")?;

        let epoch = system_state.epoch() as i64;
        let ep = system_state.emission_pool();

        // Access fields via V1 variant
        let (safe_mode, safe_mode_fees, safe_mode_emissions, protocol_fund_balance) =
            match &system_state {
                SystemState::V1(v1) => (
                    v1.safe_mode,
                    v1.safe_mode_accumulated_fees as i64,
                    v1.safe_mode_accumulated_emissions as i64,
                    v1.protocol_fund_balance as i64,
                ),
            };

        Ok(vec![StoredEpochState {
            epoch,
            emission_balance: ep.balance as i64,
            emission_per_epoch: ep.current_distribution_amount as i64,
            distribution_counter: ep.distribution_counter as i64,
            period_length: ep.period_length as i64,
            decrease_rate: ep.decrease_rate as i32,
            protocol_fund_balance,
            safe_mode,
            safe_mode_accumulated_fees: safe_mode_fees,
            safe_mode_accumulated_emissions: safe_mode_emissions,
        }])
    }
}

#[async_trait]
impl Handler for SomaEpochState {
    const MIN_EAGER_ROWS: usize = 1;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_epoch_state::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
