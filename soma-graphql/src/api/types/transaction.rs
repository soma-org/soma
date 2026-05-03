// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use ::types::effects::{ExecutionStatus, TransactionEffects, TransactionEffectsAPI};
use ::types::transaction::TransactionKind;
use async_graphql::*;

use crate::api::scalars::{Base64, BigInt, DateTime, Digest, SomaAddress};

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
        TransactionKind::BalanceTransfer(_) => "BalanceTransfer",
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
        TransactionKind::AddStake { validator, amount } => {
            Some(format!(
                r#"{{"validator":"0x{}","amount":{}}}"#,
                hex::encode(validator.to_vec()),
                amount,
            ))
        }
        _ => None,
    }
}

/// A transaction on the Soma network.
pub struct Transaction {
    pub tx_digest: Vec<u8>,
    pub cp_sequence_number: i64,
    pub timestamp_ms: i64,
    pub raw_transaction_bcs: Vec<u8>,
    pub raw_effects_bcs: Vec<u8>,
    pub user_signatures_bcs: Vec<u8>,
}

#[Object]
impl Transaction {
    /// The transaction digest (base58).
    async fn digest(&self) -> Digest {
        Digest(self.tx_digest.clone())
    }

    /// The checkpoint that included this transaction.
    async fn checkpoint_sequence_number(&self) -> BigInt {
        BigInt(self.cp_sequence_number)
    }

    /// When this transaction was included in a checkpoint.
    async fn timestamp(&self) -> DateTime {
        DateTime(self.timestamp_ms)
    }

    /// BCS-serialized transaction data.
    async fn raw_transaction_bcs(&self) -> Base64 {
        Base64(self.raw_transaction_bcs.clone())
    }

    /// BCS-serialized transaction effects.
    async fn raw_effects_bcs(&self) -> Base64 {
        Base64(self.raw_effects_bcs.clone())
    }

    /// BCS-serialized user signatures.
    async fn user_signatures_bcs(&self) -> Base64 {
        Base64(self.user_signatures_bcs.clone())
    }

    // --- Decoded fields (on-demand BCS deserialization) ---

    /// The decoded transaction kind label (e.g. "SubmitData", "CreateModel", "AddStake").
    async fn kind(&self) -> Result<String> {
        let tx: ::types::transaction::Transaction = bcs::from_bytes(&self.raw_transaction_bcs)
            .map_err(|e| Error::new(format!("BCS decode error: {e}")))?;
        let kind = tx.data().intent_message().value.kind();
        Ok(kind_label(kind).to_string())
    }

    /// The address of the transaction sender.
    async fn sender(&self) -> Result<SomaAddress> {
        let tx: ::types::transaction::Transaction = bcs::from_bytes(&self.raw_transaction_bcs)
            .map_err(|e| Error::new(format!("BCS decode error: {e}")))?;
        let sender = tx.data().intent_message().value.sender();
        Ok(SomaAddress(sender.to_vec()))
    }

    /// The epoch in which this transaction was executed.
    async fn epoch(&self) -> Result<BigInt> {
        let effects: TransactionEffects = bcs::from_bytes(&self.raw_effects_bcs)
            .map_err(|e| Error::new(format!("BCS decode error: {e}")))?;
        Ok(BigInt(effects.executed_epoch() as i64))
    }

    /// Execution status: "Success" or "Failure".
    async fn status(&self) -> Result<String> {
        let effects: TransactionEffects = bcs::from_bytes(&self.raw_effects_bcs)
            .map_err(|e| Error::new(format!("BCS decode error: {e}")))?;
        Ok(match effects.status() {
            ExecutionStatus::Success => "Success".to_string(),
            ExecutionStatus::Failure { .. } => "Failure".to_string(),
        })
    }

    /// Total gas fee deducted.
    async fn gas_used(&self) -> Result<BigInt> {
        let effects: TransactionEffects = bcs::from_bytes(&self.raw_effects_bcs)
            .map_err(|e| Error::new(format!("BCS decode error: {e}")))?;
        Ok(BigInt(effects.transaction_fee().total_fee as i64))
    }

    /// Transaction digests this transaction depends on.
    async fn dependencies(&self) -> Result<Vec<Digest>> {
        let effects: TransactionEffects = bcs::from_bytes(&self.raw_effects_bcs)
            .map_err(|e| Error::new(format!("BCS decode error: {e}")))?;
        Ok(effects
            .dependencies()
            .iter()
            .map(|d: &::types::digests::TransactionDigest| Digest(d.inner().to_vec()))
            .collect())
    }

    /// Kind-specific metadata as JSON (e.g. target_id, model_id, amount).
    async fn metadata_json(&self) -> Result<Option<String>> {
        let tx: ::types::transaction::Transaction = bcs::from_bytes(&self.raw_transaction_bcs)
            .map_err(|e| Error::new(format!("BCS decode error: {e}")))?;
        let kind = tx.data().intent_message().value.kind();
        Ok(metadata_json(kind))
    }
}
