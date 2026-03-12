// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use diesel::prelude::*;
use soma_field_count::FieldCount;

use crate::schema::soma_models;
use crate::schema::soma_rewards;
use crate::schema::soma_targets;

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_targets)]
#[diesel(treat_none_as_default_value = false)]
pub struct StoredTarget {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub epoch: i64,
    pub status: String,
    pub submitter: Option<Vec<u8>>,
    pub winning_model_id: Option<Vec<u8>>,
    pub reward_pool: i64,
    pub bond_amount: i64,
    pub report_count: i32,
    pub state_bcs: Vec<u8>,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_models)]
pub struct StoredModel {
    pub model_id: Vec<u8>,
    pub epoch: i64,
    pub status: String,
    pub owner: Vec<u8>,
    pub architecture_version: i64,
    pub commit_epoch: i64,
    pub stake: i64,
    pub commission_rate: i64,
    pub has_embedding: bool,
    pub state_bcs: Vec<u8>,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_rewards)]
pub struct StoredReward {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub epoch: i64,
    pub tx_digest: Vec<u8>,
    pub balance_changes_bcs: Vec<u8>,
}
