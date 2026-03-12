// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Transactions table: stores transaction data indexed by digest.

use anyhow::{Context, Result};
use bytes::Bytes;
use types::digests::TransactionDigest;
use types::effects::TransactionEffects;
use types::checkpoints::CheckpointSequenceNumber;
use types::transaction::Transaction;

use crate::TransactionData;

pub mod col {
    pub const EFFECTS: &str = "ef";
    pub const TIMESTAMP: &str = "ts";
    pub const CHECKPOINT_NUMBER: &str = "cn";
    pub const DATA: &str = "td";
    pub const SIGNATURES: &str = "sg";
    pub const BALANCE_CHANGES: &str = "bc";
}

pub const NAME: &str = "transactions";

pub fn encode_key(digest: &TransactionDigest) -> Vec<u8> {
    digest.inner().to_vec()
}

/// Encode all transaction columns.
/// Soma has no events, so we skip the events column entirely.
pub fn encode(
    transaction: &Transaction,
    effects: &TransactionEffects,
    checkpoint_number: CheckpointSequenceNumber,
    timestamp_ms: u64,
    balance_changes_bcs: &[u8],
) -> Result<Vec<(&'static str, Bytes)>> {
    Ok(vec![
        (col::EFFECTS, Bytes::from(bcs::to_bytes(effects)?)),
        (col::TIMESTAMP, Bytes::from(bcs::to_bytes(&timestamp_ms)?)),
        (
            col::CHECKPOINT_NUMBER,
            Bytes::from(bcs::to_bytes(&checkpoint_number)?),
        ),
        (
            col::DATA,
            Bytes::from(bcs::to_bytes(&transaction.data().intent_message().value)?),
        ),
        (
            col::SIGNATURES,
            Bytes::from(bcs::to_bytes(transaction.data().tx_signatures())?),
        ),
        (col::BALANCE_CHANGES, Bytes::from(balance_changes_bcs.to_vec())),
    ])
}

pub fn decode(row: &[(Bytes, Bytes)]) -> Result<TransactionData> {
    let mut tx_data: Option<types::transaction::TransactionData> = None;
    let mut tx_signatures = None;
    let mut effects = None;
    let mut timestamp = 0;
    let mut checkpoint_number = 0;

    for (column, value) in row {
        match column.as_ref() {
            b"td" => tx_data = Some(bcs::from_bytes(value)?),
            b"sg" => tx_signatures = Some(bcs::from_bytes(value)?),
            b"ef" => effects = Some(bcs::from_bytes(value)?),
            b"ts" => timestamp = bcs::from_bytes(value)?,
            b"cn" => checkpoint_number = bcs::from_bytes(value)?,
            _ => {}
        }
    }

    let transaction = {
        let data = tx_data.context("transaction data field is missing")?;
        let sigs = tx_signatures.context("transaction signatures field is missing")?;
        let sender_signed_data = types::transaction::SenderSignedData::new(data, sigs);
        Transaction::new(sender_signed_data)
    };

    Ok(TransactionData {
        transaction,
        effects: effects.context("effects field is missing")?,
        timestamp,
        checkpoint_number,
    })
}
