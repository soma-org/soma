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
        TransactionKind::TransferCoin { .. } => "TransferCoin",
        TransactionKind::PayCoins { .. } => "PayCoins",
        TransactionKind::TransferObjects { .. } => "TransferObjects",
        TransactionKind::AddStake { .. } => "AddStake",
        TransactionKind::WithdrawStake { .. } => "WithdrawStake",
        TransactionKind::CreateModel(_) => "CreateModel",
        TransactionKind::CommitModel(_) => "CommitModel",
        TransactionKind::RevealModel(_) => "RevealModel",
        TransactionKind::AddStakeToModel { .. } => "AddStakeToModel",
        TransactionKind::SetModelCommissionRate { .. } => "SetModelCommissionRate",
        TransactionKind::DeactivateModel { .. } => "DeactivateModel",
        TransactionKind::ReportModel { .. } => "ReportModel",
        TransactionKind::UndoReportModel { .. } => "UndoReportModel",
        TransactionKind::SubmitData(_) => "SubmitData",
        TransactionKind::ClaimRewards(_) => "ClaimRewards",
        TransactionKind::ReportSubmission { .. } => "ReportSubmission",
        TransactionKind::UndoReportSubmission { .. } => "UndoReportSubmission",
    }
}

/// Extract kind-specific metadata as a JSON string for interesting tx types.
fn metadata_json(kind: &TransactionKind) -> Option<String> {
    match kind {
        TransactionKind::SubmitData(args) => Some(format!(
            r#"{{"target_id":"0x{}","model_id":"0x{}"}}"#,
            hex::encode(args.target_id.to_vec()),
            hex::encode(args.model_id.to_vec()),
        )),
        TransactionKind::CommitModel(args) => {
            Some(format!(r#"{{"model_id":"0x{}"}}"#, hex::encode(args.model_id.to_vec()),))
        }
        TransactionKind::RevealModel(args) => {
            Some(format!(r#"{{"model_id":"0x{}"}}"#, hex::encode(args.model_id.to_vec()),))
        }
        TransactionKind::ClaimRewards(args) => {
            Some(format!(r#"{{"target_id":"0x{}"}}"#, hex::encode(args.target_id.to_vec()),))
        }
        TransactionKind::AddStakeToModel { model_id, amount, .. } => {
            let amount_str = match amount {
                Some(a) => format!("{a}"),
                None => "null".to_string(),
            };
            Some(format!(
                r#"{{"model_id":"0x{}","amount":{}}}"#,
                hex::encode(model_id.to_vec()),
                amount_str,
            ))
        }
        TransactionKind::SetModelCommissionRate { model_id, new_rate } => Some(format!(
            r#"{{"model_id":"0x{}","new_rate":{}}}"#,
            hex::encode(model_id.to_vec()),
            new_rate,
        )),
        TransactionKind::DeactivateModel { model_id } => {
            Some(format!(r#"{{"model_id":"0x{}"}}"#, hex::encode(model_id.to_vec()),))
        }
        TransactionKind::ReportModel { model_id } => {
            Some(format!(r#"{{"model_id":"0x{}"}}"#, hex::encode(model_id.to_vec()),))
        }
        TransactionKind::UndoReportModel { model_id } => {
            Some(format!(r#"{{"model_id":"0x{}"}}"#, hex::encode(model_id.to_vec()),))
        }
        TransactionKind::ReportSubmission { target_id } => {
            Some(format!(r#"{{"target_id":"0x{}"}}"#, hex::encode(target_id.to_vec()),))
        }
        TransactionKind::UndoReportSubmission { target_id } => {
            Some(format!(r#"{{"target_id":"0x{}"}}"#, hex::encode(target_id.to_vec()),))
        }
        TransactionKind::TransferCoin { amount, recipient, .. } => {
            let amount_str = match amount {
                Some(a) => format!("{a}"),
                None => "null".to_string(),
            };
            Some(format!(
                r#"{{"recipient":"0x{}","amount":{}}}"#,
                hex::encode(recipient.to_vec()),
                amount_str,
            ))
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
