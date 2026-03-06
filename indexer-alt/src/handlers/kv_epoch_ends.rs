// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use async_trait::async_trait;
use diesel::ExpressionMethods;
use diesel::QueryDsl;
use diesel_async::RunQueryDsl;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::transaction::TransactionKind;
use indexer_alt_schema::epochs::StoredEpochEnd;
use indexer_alt_schema::schema::kv_epoch_ends;

use crate::handlers::cp_sequence_numbers::epoch_interval;

pub struct KvEpochEnds;

#[async_trait]
impl Processor for KvEpochEnds {
    const NAME: &'static str = "kv_epoch_ends";

    type Value = StoredEpochEnd;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint {
            summary,
            transactions,
            ..
        } = checkpoint.as_ref();

        let Some(end_of_epoch) = summary.end_of_epoch_data.as_ref() else {
            return Ok(vec![]);
        };

        // Find the ChangeEpoch transaction
        let Some(_transaction) = transactions.iter().find(|tx| {
            matches!(tx.transaction.kind(), TransactionKind::ChangeEpoch(_))
        }) else {
            bail!(
                "Failed to get ChangeEpoch transaction in checkpoint {} with EndOfEpochData",
                summary.sequence_number,
            );
        };

        // Soma does not have SystemEpochInfoEvent or events at all.
        // We record the epoch end with the data available from the checkpoint summary.
        // The detailed staking/storage fields are set to None (safe_mode = true style)
        // since Soma's epoch transitions don't emit the same event data as Sui.
        Ok(vec![StoredEpochEnd {
            epoch: summary.epoch as i64,
            cp_hi: summary.sequence_number as i64 + 1,
            tx_hi: summary.network_total_transactions as i64,
            end_timestamp_ms: summary.timestamp_ms as i64,

            safe_mode: false,

            // Soma does not have SystemEpochInfoEvent, so these fields are not
            // available from events. They could be extracted from system state
            // if needed in the future.
            total_stake: None,
            storage_fund_balance: None,
            storage_fund_reinvestment: None,
            storage_charge: None,
            storage_rebate: None,
            stake_subsidy_amount: None,
            total_gas_fees: None,
            total_stake_rewards_distributed: None,
            leftover_storage_fund_inflow: None,

            epoch_commitments: bcs::to_bytes(&end_of_epoch.epoch_commitments)
                .context("Failed to serialize EpochCommitment-s")?,
        }])
    }
}

#[async_trait]
impl Handler for KvEpochEnds {
    const MIN_EAGER_ROWS: usize = 1;

    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(kv_epoch_ends::table)
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
        let Range {
            start: from_epoch,
            end: to_epoch,
        } = epoch_interval(conn, from..to_exclusive).await?;
        if from_epoch < to_epoch {
            let filter = kv_epoch_ends::table
                .filter(kv_epoch_ends::epoch.between(from_epoch as i64, to_epoch as i64 - 1));
            Ok(diesel::delete(filter).execute(conn).await?)
        } else {
            Ok(0)
        }
    }
}
