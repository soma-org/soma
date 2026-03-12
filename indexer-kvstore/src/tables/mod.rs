// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Table schema definitions with encode/decode functions for BigTable.

use bytes::Bytes;

use crate::bigtable::proto::bigtable::v2::mutate_rows_request::Entry;
use crate::bigtable::proto::bigtable::v2::mutation::SetCell;
use crate::bigtable::proto::bigtable::v2::{Mutation, mutation};

pub mod checkpoints;
pub mod checkpoints_by_digest;
pub mod epochs;
pub mod objects;
pub mod transactions;
pub mod watermarks;

// Soma-specific tables
pub mod targets;
pub mod models;
pub mod rewards;

/// Column family name used by all tables.
pub const FAMILY: &str = "soma";

/// Default column qualifier for single-column tables.
pub const DEFAULT_COLUMN: &str = "";

pub mod watermark_alt_legacy {
    pub const NAME: &str = "watermark_alt";
}

/// Build an Entry from cells.
pub fn make_entry(
    row_key: impl Into<Bytes>,
    cells: impl IntoIterator<Item = (&'static str, Bytes)>,
    timestamp_ms: Option<u64>,
) -> Entry {
    let timestamp_micros = timestamp_ms
        .map(|ms| ms.checked_mul(1000).expect("timestamp overflow") as i64)
        .unwrap_or(-1);

    Entry {
        row_key: row_key.into(),
        mutations: cells
            .into_iter()
            .map(|(col, val)| Mutation {
                mutation: Some(mutation::Mutation::SetCell(SetCell {
                    family_name: FAMILY.to_string(),
                    column_qualifier: Bytes::from(col),
                    timestamp_micros,
                    value: val,
                })),
            })
            .collect(),
        idempotency: None,
    }
}
