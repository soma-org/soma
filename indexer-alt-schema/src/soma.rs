// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use diesel::prelude::*;
use soma_field_count::FieldCount;

use crate::schema::soma_epoch_state;
use crate::schema::soma_models;
use crate::schema::soma_reward_balances;
use crate::schema::soma_rewards;
use crate::schema::soma_staked_soma;
use crate::schema::soma_target_reports;
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
    pub winning_distance_score: Option<f64>,
    pub winning_loss_score: Option<f64>,
    pub winning_model_owner: Option<Vec<u8>>,
    pub fill_epoch: Option<i64>,
    pub distance_threshold: f64,
    pub model_ids_json: String,
    pub winning_data_url: Option<String>,
    pub winning_data_checksum: Option<Vec<u8>>,
    pub winning_data_size: Option<i64>,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_models)]
#[diesel(treat_none_as_default_value = false)]
pub struct StoredModel {
    pub model_id: Vec<u8>,
    pub epoch: i64,
    pub status: String,
    pub owner: Vec<u8>,
    pub architecture_version: i64,
    pub commit_epoch: i64,
    pub stake: i64,
    pub commission_rate: i64,
    pub next_epoch_commission_rate: i64,
    pub staking_pool_id: Vec<u8>,
    pub activation_epoch: Option<i64>,
    pub deactivation_epoch: Option<i64>,
    pub rewards_pool: i64,
    pub pool_token_balance: i64,
    pub pending_stake: i64,
    pub pending_total_soma_withdraw: i64,
    pub pending_pool_token_withdraw: i64,
    pub exchange_rates_json: String,
    pub manifest_url: Option<String>,
    pub manifest_checksum: Option<Vec<u8>>,
    pub manifest_size: Option<i64>,
    pub weights_commitment: Option<Vec<u8>>,
    pub has_pending_update: bool,
    pub pending_manifest_url: Option<String>,
    pub pending_manifest_checksum: Option<Vec<u8>>,
    pub pending_manifest_size: Option<i64>,
    pub pending_weights_commitment: Option<Vec<u8>>,
    pub pending_commit_epoch: Option<i64>,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_staked_soma)]
#[diesel(treat_none_as_default_value = false)]
pub struct StoredStakedSoma {
    pub staked_soma_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub owner: Option<Vec<u8>>,
    pub pool_id: Option<Vec<u8>>,
    pub stake_activation_epoch: Option<i64>,
    pub principal: Option<i64>,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_epoch_state)]
pub struct StoredEpochState {
    pub epoch: i64,
    pub emission_balance: i64,
    pub emission_per_epoch: i64,
    pub distance_threshold: f64,
    pub targets_generated_this_epoch: i64,
    pub hits_this_epoch: i64,
    pub hits_ema: i64,
    pub reward_per_target: i64,
    pub safe_mode: bool,
    pub safe_mode_accumulated_fees: i64,
    pub safe_mode_accumulated_emissions: i64,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_target_reports)]
pub struct StoredTargetReport {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub reporter: Vec<u8>,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_rewards)]
pub struct StoredReward {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub epoch: i64,
    pub tx_digest: Vec<u8>,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_reward_balances)]
pub struct StoredRewardBalance {
    pub target_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub epoch: i64,
    pub tx_digest: Vec<u8>,
    pub recipient: Vec<u8>,
    pub amount: i64,
}
