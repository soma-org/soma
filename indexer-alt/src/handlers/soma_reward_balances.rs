// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_reward_balances;
use indexer_alt_schema::soma::StoredRewardBalance;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::balance_change::derive_balance_changes_2;
use types::full_checkpoint_content::Checkpoint;
use types::transaction::TransactionKind;

pub struct SomaRewardBalances;

#[async_trait]
impl Processor for SomaRewardBalances {
    const NAME: &'static str = "soma_reward_balances";

    type Value = StoredRewardBalance;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let cp_sequence_number = checkpoint.summary.sequence_number as i64;
        let mut values = vec![];

        for tx in &checkpoint.transactions {
            let TransactionKind::ClaimRewards(args) = tx.transaction.kind() else {
                continue;
            };

            let changes = derive_balance_changes_2(&tx.effects, &checkpoint.object_set);
            let target_id = args.target_id.to_vec();
            let epoch = checkpoint.summary.epoch as i64;
            let tx_digest = tx.transaction.digest().inner().to_vec();

            for change in changes {
                values.push(StoredRewardBalance {
                    target_id: target_id.clone(),
                    cp_sequence_number,
                    epoch,
                    tx_digest: tx_digest.clone(),
                    recipient: change.address.to_vec(),
                    amount: change.amount as i64,
                });
            }
        }
        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaRewardBalances {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_reward_balances::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
