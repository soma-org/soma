// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use diesel::prelude::*;
use soma_field_count::FieldCount;

use crate::schema::soma_asks;
use crate::schema::soma_bids;
use crate::schema::soma_epoch_state;
use crate::schema::soma_settlements;
use crate::schema::soma_staked_soma;
use crate::schema::soma_validators;
use crate::schema::soma_vaults;

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
    pub distribution_counter: i64,
    pub period_length: i64,
    pub decrease_rate: i32,
    pub protocol_fund_balance: i64,
    pub safe_mode: bool,
    pub safe_mode_accumulated_fees: i64,
    pub safe_mode_accumulated_emissions: i64,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_validators)]
#[diesel(treat_none_as_default_value = false)]
pub struct StoredValidator {
    pub address: Vec<u8>,
    pub epoch: i64,
    pub voting_power: i64,
    pub commission_rate: i64,
    pub next_epoch_commission_rate: i64,
    pub staking_pool_id: Vec<u8>,
    pub stake: i64,
    pub pending_stake: i64,
    pub name: Option<String>,
    pub network_address: Option<String>,
    pub proxy_address: Option<String>,
    pub protocol_pubkey: Option<Vec<u8>>,
}

// --- Marketplace stored types ---

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_asks)]
pub struct StoredAsk {
    pub ask_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub buyer: Vec<u8>,
    pub task_digest: Vec<u8>,
    pub max_price_per_bid: i64,
    pub num_bids_wanted: i32,
    pub timeout_ms: i64,
    pub created_at_ms: i64,
    pub status: String,
    pub accepted_bid_count: i32,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_bids)]
pub struct StoredBid {
    pub bid_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub ask_id: Vec<u8>,
    pub seller: Vec<u8>,
    pub price: i64,
    pub response_digest: Vec<u8>,
    pub created_at_ms: i64,
    pub status: String,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_settlements)]
pub struct StoredSettlement {
    pub settlement_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub ask_id: Vec<u8>,
    pub bid_id: Vec<u8>,
    pub buyer: Vec<u8>,
    pub seller: Vec<u8>,
    pub amount: i64,
    pub task_digest: Vec<u8>,
    pub response_digest: Vec<u8>,
    pub settled_at_ms: i64,
    pub seller_rating: String,
    pub rating_deadline_ms: i64,
}

#[derive(Insertable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_vaults)]
pub struct StoredVault {
    pub vault_id: Vec<u8>,
    pub cp_sequence_number: i64,
    pub owner: Vec<u8>,
    pub balance: i64,
}
