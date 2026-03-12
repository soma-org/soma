// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Models table: stores model snapshots indexed by (model_id, epoch).

use anyhow::{Context, Result};
use bytes::Bytes;

use crate::tables::DEFAULT_COLUMN;

pub const NAME: &str = "models";

/// Key: model_id bytes (32) + epoch (8 BE)
pub fn encode_key(model_id: &[u8], epoch: u64) -> Vec<u8> {
    let mut key = model_id.to_vec();
    key.extend(epoch.to_be_bytes());
    key
}

/// Stores full BCS-serialized ModelV1.
pub fn encode(state_bcs: &[u8]) -> [(&'static str, Bytes); 1] {
    [(DEFAULT_COLUMN, Bytes::from(state_bcs.to_vec()))]
}

pub fn decode(row: &[(Bytes, Bytes)]) -> Result<Vec<u8>> {
    let (_, value) = row.first().context("empty row")?;
    Ok(value.to_vec())
}
