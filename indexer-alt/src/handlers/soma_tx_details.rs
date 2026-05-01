// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_tx_details;
use indexer_alt_schema::transactions::StoredTxDetail;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::full_checkpoint_content::Checkpoint;
use types::transaction::TransactionKind;

pub struct SomaTxDetails;

/// Map a `TransactionKind` to a human-readable label string.
fn kind_label(kind: &TransactionKind) -> &'static str {
    match kind {
        TransactionKind::Genesis(_) => "Genesis",
        TransactionKind::ConsensusCommitPrologueV1(_) => "ConsensusCommitPrologue",
        TransactionKind::ChangeEpoch(_) => "ChangeEpoch",
        TransactionKind::AddValidator(_) => "AddValidator",
        TransactionKind::RemoveValidator(_) => "RemoveValidator",
        TransactionKind::ReportValidator { .. } => "ReportValidator",
        TransactionKind::UndoReportValidator { .. } => "UndoReportValidator",
        TransactionKind::UpdateValidatorMetadata(_) => "UpdateValidatorMetadata",
        TransactionKind::SetCommissionRate { .. } => "SetCommissionRate",
        TransactionKind::Transfer { .. } => "Transfer",
        TransactionKind::MergeCoins { .. } => "MergeCoins",
        TransactionKind::TransferObjects { .. } => "TransferObjects",
        TransactionKind::AddStake { .. } => "AddStake",
        TransactionKind::WithdrawStake { .. } => "WithdrawStake",
        TransactionKind::BridgeDeposit(_) => "BridgeDeposit",
        TransactionKind::BridgeWithdraw(_) => "BridgeWithdraw",
        TransactionKind::BridgeEmergencyPause(_) => "BridgeEmergencyPause",
        TransactionKind::BridgeEmergencyUnpause(_) => "BridgeEmergencyUnpause",
        TransactionKind::OpenChannel(_) => "OpenChannel",
        TransactionKind::Settle(_) => "Settle",
        TransactionKind::RequestClose(_) => "RequestClose",
        TransactionKind::WithdrawAfterTimeout(_) => "WithdrawAfterTimeout",
        TransactionKind::Settlement(_) => "Settlement",
    }
}

/// Extract kind-specific metadata as a JSON string for interesting tx types.
fn metadata_json(kind: &TransactionKind) -> Option<String> {
    match kind {
        TransactionKind::Transfer { amounts, recipients, .. } => {
            let amount_str = match amounts {
                Some(a) => format!("{:?}", a),
                None => "null".to_string(),
            };
            let recipients_str: Vec<String> = recipients
                .iter()
                .map(|r| format!("\"0x{}\"", hex::encode(r.to_vec())))
                .collect();
            Some(format!(
                r#"{{"recipients":[{}],"amounts":{}}}"#,
                recipients_str.join(","),
                amount_str,
            ))
        }
        TransactionKind::MergeCoins { coins } => {
            Some(format!(r#"{{"coin_count":{}}}"#, coins.len()))
        }
        TransactionKind::AddStake { address, amount, .. } => {
            let amount_str = match amount {
                Some(a) => format!("{a}"),
                None => "null".to_string(),
            };
            Some(format!(
                r#"{{"address":"0x{}","amount":{}}}"#,
                hex::encode(address.to_vec()),
                amount_str,
            ))
        }
        _ => None,
    }
}

#[async_trait]
impl Processor for SomaTxDetails {
    const NAME: &'static str = "soma_tx_details";

    type Value = StoredTxDetail;

    async fn process(&self, checkpoint: &Arc<Checkpoint>) -> Result<Vec<Self::Value>> {
        let Checkpoint { transactions, summary, .. } = checkpoint.as_ref();

        let cp_sequence_number = summary.sequence_number as i64;
        let timestamp_ms = summary.timestamp_ms as i64;
        let first_tx = summary.network_total_transactions as usize - transactions.len();

        // Determine epoch from cp_sequence_numbers or summary
        let epoch = summary.epoch as i64;

        let mut values = Vec::with_capacity(transactions.len());
        for (i, tx) in transactions.iter().enumerate() {
            let tx_sequence_number = (first_tx + i) as i64;
            let kind = tx.transaction.kind();
            let sender = tx.transaction.sender();

            values.push(StoredTxDetail {
                tx_sequence_number,
                tx_digest: tx.transaction.digest().inner().to_vec(),
                kind: kind_label(kind).to_string(),
                sender: sender.to_vec(),
                epoch,
                timestamp_ms,
                metadata_json: metadata_json(kind),
            });
        }

        Ok(values)
    }
}

#[async_trait]
impl Handler for SomaTxDetails {
    async fn commit<'a>(values: &[Self::Value], conn: &mut Connection<'a>) -> Result<usize> {
        Ok(diesel::insert_into(soma_tx_details::table)
            .values(values)
            .on_conflict_do_nothing()
            .execute(conn)
            .await?)
    }
}
