// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Objects table: stores object data indexed by (ObjectID, version).

use anyhow::{Context, Result};
use bytes::Bytes;
use types::object::Object;
use types::storage::ObjectKey;

use crate::tables::DEFAULT_COLUMN;

pub const NAME: &str = "objects";

pub fn encode_key(object_key: &ObjectKey) -> Vec<u8> {
    let mut raw_key = object_key.0.to_vec();
    raw_key.extend(object_key.1.value().to_be_bytes());
    raw_key
}

pub fn encode(object: &Object) -> Result<[(&'static str, Bytes); 1]> {
    Ok([(DEFAULT_COLUMN, Bytes::from(bcs::to_bytes(object)?))])
}

pub fn decode(row: &[(Bytes, Bytes)]) -> Result<Object> {
    let (_, value) = row.first().context("empty row")?;
    Ok(bcs::from_bytes(value)?)
}
