// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Rewards table: stores reward claim data indexed by (target_id, tx_digest).

use anyhow::{Context, Result};
use bytes::Bytes;

use crate::tables::DEFAULT_COLUMN;

pub const NAME: &str = "rewards";

/// Key: target_id bytes (32) + tx_digest bytes (32)
pub fn encode_key(target_id: &[u8], tx_digest: &[u8]) -> Vec<u8> {
    let mut key = target_id.to_vec();
    key.extend(tx_digest);
    key
}

/// Stores BCS-serialized balance changes.
pub fn encode(balance_changes_bcs: &[u8]) -> [(&'static str, Bytes); 1] {
    [(DEFAULT_COLUMN, Bytes::from(balance_changes_bcs.to_vec()))]
}

pub fn decode(row: &[(Bytes, Bytes)]) -> Result<Vec<u8>> {
    let (_, value) = row.first().context("empty row")?;
    Ok(value.to_vec())
}
