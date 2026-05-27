// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use diesel_async::RunQueryDsl;
use indexer_alt_schema::schema::soma_tx_details;
use indexer_alt_schema::transactions::StoredTxDetail;
use indexer_framework::pipeline::Processor;
use indexer_framework::postgres::Connection;
use indexer_framework::postgres::handler::Handler;
use types::balance::BalanceEvent;
use types::base::SomaAddress;
use types::full_checkpoint_content::Checkpoint;
use types::transaction::{SettlementTransaction, TransactionKind};

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
        // Stage 13b: Transfer / MergeCoins variants gone.
        TransactionKind::TransferObjects { .. } => "TransferObjects",
        TransactionKind::AddStake { .. } => "AddStake",
        TransactionKind::WithdrawStake { .. } => "WithdrawStake",
        TransactionKind::BridgeDeposit(_) => "BridgeDeposit",
        TransactionKind::BridgeWithdraw(_) => "BridgeWithdraw",
        TransactionKind::BridgeEmergencyPause(_) => "BridgeEmergencyPause",
        TransactionKind::BridgeEmergencyUnpause(_) => "BridgeEmergencyUnpause",
        TransactionKind::BridgeAttachWithdrawalSignatures(_) => "BridgeAttachWithdrawalSignatures",
        TransactionKind::BridgeUpdateCommitteeBlocklist(_) => "BridgeUpdateCommitteeBlocklist",
        TransactionKind::BridgeRegisterBridgeKey(_) => "BridgeRegisterBridgeKey",
        TransactionKind::OpenChannel(_) => "OpenChannel",
        TransactionKind::Settle(_) => "Settle",
        TransactionKind::RequestClose(_) => "RequestClose",
        TransactionKind::WithdrawAfterTimeout(_) => "WithdrawAfterTimeout",
        TransactionKind::TopUp(_) => "TopUp",
        TransactionKind::RateChannel(_) => "RateChannel",
        TransactionKind::RegisterProvider(_) => "RegisterProvider",
        TransactionKind::UpdateProvider(_) => "UpdateProvider",
        TransactionKind::RegisterOffering(_) => "RegisterOffering",
        TransactionKind::UpdateOffering(_) => "UpdateOffering",
        TransactionKind::DeactivateOffering(_) => "DeactivateOffering",
        TransactionKind::Settlement(_) => "Settlement",
        TransactionKind::BalanceTransfer(_) => "BalanceTransfer",
    }
}

/// Extract kind-specific metadata as a JSON string for interesting tx types.
fn metadata_json(kind: &TransactionKind) -> Option<String> {
    match kind {
        // Stage 13b: Transfer / MergeCoins variants gone.
        TransactionKind::AddStake { validator, amount } => Some(format!(
            r#"{{"validator":"0x{}","amount":{}}}"#,
            hex::encode(validator.to_vec()),
            amount,
        )),
        // Settlement is system-authored (sender = ZERO by executor
        // invariant — see authority/src/execution/settlement.rs). The
        // economically meaningful actors are the BalanceEvent owners;
        // surface the aggregated per-(owner, coin_type) net deltas so
        // explorer + downstream consumers can attribute the tx.
        TransactionKind::Settlement(s) => settlement_metadata_json(s),
        _ => None,
    }
}

/// Aggregate Settlement balance/delegation changes into a compact
/// JSON summary for `metadata_json`. Keys are per-(owner, coin_type)
/// signed net deltas (i128 stringified, since the absolute amount can
/// exceed i64 in degenerate cases).
fn settlement_metadata_json(s: &SettlementTransaction) -> Option<String> {
    if s.changes.is_empty() && s.delegation_changes.is_empty() {
        return None;
    }
    let deltas = aggregate_balance_changes(&s.changes);
    let mut parts: Vec<String> = Vec::new();
    for ((owner, coin), delta) in &deltas {
        parts.push(format!(
            r#"{{"owner":"0x{}","coin_type":"{:?}","delta":"{}"}}"#,
            hex::encode(owner.to_vec()),
            coin,
            delta,
        ));
    }
    Some(format!(
        r#"{{"changes":[{}],"delegation_change_count":{}}}"#,
        parts.join(","),
        s.delegation_changes.len(),
    ))
}

/// Net signed delta per `(owner, coin_type)` across a Settlement's
/// `BalanceEvent` list. Used both to pick a representative `sender`
/// and to build the metadata JSON. Mirrors `balance::aggregate_events`
/// but keeps the `i128` deltas (positive = net deposit, negative =
/// net withdraw).
fn aggregate_balance_changes(
    changes: &[BalanceEvent],
) -> BTreeMap<(SomaAddress, types::object::CoinType), i128> {
    let mut out: BTreeMap<(SomaAddress, types::object::CoinType), i128> = BTreeMap::new();
    for e in changes {
        *out.entry(e.aggregation_key()).or_insert(0) += e.signed_delta();
    }
    out
}

/// Pick a representative sender for a `Settlement`. The on-chain
/// sender is always `SomaAddress::ZERO` because the system address
/// is the only valid signer (see settlement executor). For explorer
/// and `transactions(sender:filter)` semantics we want the address
/// that actually moved value — the unique BalanceEvent owner when
/// the cp aggregated only one party's activity, otherwise ZERO
/// (multi-party cps have no single "submitter").
fn settlement_representative_sender(kind: &TransactionKind) -> Option<SomaAddress> {
    let TransactionKind::Settlement(s) = kind else {
        return None;
    };
    let mut owners = std::collections::HashSet::new();
    for e in &s.changes {
        owners.insert(e.owner());
        if owners.len() > 1 {
            return None;
        }
    }
    owners.into_iter().next()
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
            // For Settlement we override the on-chain ZERO sender
            // with the dominant BalanceEvent owner so the explorer
            // attributes the tx to the address that moved value (the
            // common single-Settle-per-cp case). Multi-party cps stay
            // ZERO — no single submitter exists.
            let sender =
                settlement_representative_sender(kind).unwrap_or_else(|| tx.transaction.sender());

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
