// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use indexer_framework::pipeline::Processor;
use types::balance_change::derive_balance_changes_2;
use types::full_checkpoint_content::Checkpoint;
use types::transaction::TransactionKind;

use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::handlers::BigTableProcessor;
use crate::tables;

/// KV pipeline that writes reward claims to BigTable.
pub struct SomaRewardsPipeline;

#[async_trait::async_trait]
impl Processor for SomaRewardsPipeline {
    const NAME: &'static str = "kvstore_soma_rewards";
    type Value = Entry;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> anyhow::Result<Vec<Self::Value>> {
        let timestamp_ms = checkpoint.summary.timestamp_ms;
        let mut entries = vec![];

        for tx in &checkpoint.transactions {
            let TransactionKind::ClaimRewards(args) = tx.transaction.kind() else {
                continue;
            };

            let changes = derive_balance_changes_2(&tx.effects, &checkpoint.object_set);
            let balance_changes_bcs = bcs::to_bytes(&changes)?;

            // Build a Transaction wrapper to get the digest
            let transaction = types::transaction::Transaction::from_generic_sig_data(
                tx.transaction.clone(),
                tx.signatures.clone(),
            );
            let tx_digest = transaction.digest().inner().to_vec();

            let entry = tables::make_entry(
                tables::rewards::encode_key(&args.target_id.to_vec(), &tx_digest),
                tables::rewards::encode(&balance_changes_bcs),
                Some(timestamp_ms),
            );
            entries.push(entry);
        }

        Ok(entries)
    }
}

impl BigTableProcessor for SomaRewardsPipeline {
    const TABLE: &'static str = tables::rewards::NAME;
}
