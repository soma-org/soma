// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use diesel::prelude::*;
use soma_field_count::FieldCount;

use crate::schema::soma_balance_deltas;
use crate::schema::soma_channel_events;
use crate::schema::soma_channel_ratings;
use crate::schema::soma_channels;
use crate::schema::soma_epoch_state;
use crate::schema::soma_providers;
use crate::schema::soma_staked_soma;
use crate::schema::soma_validators;

/// Stage 13m: per-checkpoint signed balance delta for one
/// `(owner, coin_type)`. The pipeline emits one row per
/// `(owner, coin_type, cp_sequence_number)` whose contents differ
/// (i.e. some tx in the cp touched that balance). Current balance is
/// the SUM of `delta` over all rows for the `(owner, coin_type)`.
#[derive(Insertable, Debug, Clone, FieldCount, Queryable)]
#[diesel(table_name = soma_balance_deltas)]
pub struct StoredBalanceDelta {
    pub owner: Vec<u8>,
    pub coin_type: String,
    pub cp_sequence_number: i64,
    pub delta: i64,
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
    pub protocol_pubkey: Option<Vec<u8>>,
}

// --- Channel registry ---

/// Per-channel state mirror. INSERT on `OpenChannel`, UPDATE on
/// `Settle` / `RequestClose` / `TopUp` / `WithdrawAfterTimeout`.
/// `status` follows the off-chain `ChannelStatus` enum: 0 open,
/// 1 closing, 2 withdrawn.
#[derive(Insertable, AsChangeset, Queryable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_channels)]
#[diesel(treat_none_as_null = true)]
pub struct StoredChannel {
    pub channel_id: Vec<u8>,
    pub payer: Vec<u8>,
    pub payee: Vec<u8>,
    pub authorized_signer: Vec<u8>,
    pub token: String,
    pub deposit: i64,
    pub settled_amount: i64,
    pub close_requested_at_ms: Option<i64>,
    pub status: i16,
    pub opened_at_cp: i64,
    pub opened_tx_digest: Vec<u8>,
    pub last_update_cp: i64,
}

/// Append-only event log for channel ops. One row per channel tx.
#[derive(Insertable, Queryable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_channel_events)]
pub struct StoredChannelEvent {
    pub tx_sequence_number: i64,
    pub cp_sequence_number: i64,
    pub channel_id: Vec<u8>,
    pub kind: String,
    pub delta: i64,
    pub timestamp_ms: i64,
}

/// Channel rating mirror. UPSERT on every `RateChannel` tx
/// (latest-wins). One row per rated channel. `payee` is denormalized
/// for fast per-provider aggregation in the `provider_reputation`
/// view.
#[derive(Insertable, AsChangeset, Queryable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_channel_ratings)]
pub struct StoredChannelRating {
    pub channel_id: Vec<u8>,
    pub payer: Vec<u8>,
    pub payee: Vec<u8>,
    /// `true` = thumbs down; `false` = thumbs up.
    pub negative: bool,
    pub rated_at_cp: i64,
    pub rated_at_ms: i64,
}

// --- Provider registry ---

/// On-chain provider record mirror. INSERT on `RegisterProvider`,
/// UPDATE on `UpdateProvider`.
#[derive(Insertable, AsChangeset, Queryable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_providers)]
pub struct StoredProvider {
    pub address: Vec<u8>,
    pub endpoint: String,
    pub last_update_cp: i64,
}
