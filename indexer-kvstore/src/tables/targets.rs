// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Targets table: stores target data indexed by target_id.

use anyhow::{Context, Result};
use bytes::Bytes;

use crate::tables::DEFAULT_COLUMN;

pub const NAME: &str = "targets";

/// Key: target ObjectID bytes (32 bytes)
pub fn encode_key(target_id: &[u8]) -> Vec<u8> {
    target_id.to_vec()
}

/// Stores full BCS-serialized TargetV1.
pub fn encode(state_bcs: &[u8]) -> [(&'static str, Bytes); 1] {
    [(DEFAULT_COLUMN, Bytes::from(state_bcs.to_vec()))]
}

pub fn decode(row: &[(Bytes, Bytes)]) -> Result<Vec<u8>> {
    let (_, value) = row.first().context("empty row")?;
    Ok(value.to_vec())
}
