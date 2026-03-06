// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use diesel::prelude::*;
use soma_field_count::FieldCount;

use crate::schema::kv_checkpoints;

#[derive(Insertable, Debug, Clone, FieldCount, Queryable)]
#[diesel(table_name = kv_checkpoints)]
pub struct StoredCheckpoint {
    pub sequence_number: i64,
    /// BCS serialized CheckpointContents
    pub checkpoint_contents: Vec<u8>,
    /// BCS serialized CheckpointSummary
    pub checkpoint_summary: Vec<u8>,
    /// BCS serialized AuthorityQuorumSignInfo
    pub validator_signatures: Vec<u8>,
}
