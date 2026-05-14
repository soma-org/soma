// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use diesel::prelude::*;
use soma_field_count::FieldCount;

use crate::schema::soma_balance_deltas;
use crate::schema::soma_bridge_deposits;
use crate::schema::soma_channel_events;
use crate::schema::soma_channel_ratings;
use crate::schema::soma_channels;
use crate::schema::soma_epoch_state;
use crate::schema::soma_inference_settlements;
use crate::schema::soma_offerings;
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
/// 1 closing, 2 withdrawn. 2026-05-13: now carries the
/// per-channel offering snapshot — `model_id`, prices, SLA bounds —
/// mirroring `ChannelV1`'s on-chain fields exactly.
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
    pub model_id: String,
    pub prompt_micros_per_1k: i64,
    pub completion_micros_per_1k: i64,
    pub cache_read_micros_per_1k: i64,
    pub cache_write_micros_per_1k: i64,
    pub request_micros: i64,
    pub ttft_bound_ms: i32,
    pub ttot_bound_ms: i32,
}

/// Append-only event log for channel ops. One row per channel tx.
/// 2026-05-13: usage deltas + rating reason_code populated for
/// Settle / Rate events respectively.
#[derive(Insertable, Queryable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_channel_events)]
pub struct StoredChannelEvent {
    pub tx_sequence_number: i64,
    pub cp_sequence_number: i64,
    pub channel_id: Vec<u8>,
    pub kind: String,
    pub delta: i64,
    pub timestamp_ms: i64,
    pub tokens_in_delta: i64,
    pub tokens_out_delta: i64,
    pub cache_read_delta: i64,
    pub cache_write_delta: i64,
    pub requests_delta: i64,
    pub rating_reason_code: Option<i16>,
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

// --- Bridge deposits ---

/// Per-tx materialization of `BridgeDeposit` transactions. INSERT-only;
/// every BridgeDeposit produces exactly one row. `eth_tx_hash` is the
/// Base-side L1 proof carried in `BridgeDepositArgs`.
#[derive(Insertable, Queryable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_bridge_deposits)]
pub struct StoredBridgeDeposit {
    pub tx_sequence_number: i64,
    pub cp_sequence_number: i64,
    pub recipient: Vec<u8>,
    pub amount: i64,
    pub nonce: i64,
    pub eth_tx_hash: Vec<u8>,
    pub timestamp_ms: i64,
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

// --- Per-(provider, model) offering ---

/// On-chain `Offering` mirror — one row per (provider, model_id).
/// INSERT on `RegisterOffering`, UPDATE on `UpdateOffering` /
/// `DeactivateOffering`.
#[derive(Insertable, AsChangeset, Queryable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_offerings)]
pub struct StoredOffering {
    pub provider: Vec<u8>,
    pub model_id: String,
    pub prompt_micros_per_1k: i64,
    pub completion_micros_per_1k: i64,
    pub cache_read_micros_per_1k: i64,
    pub cache_write_micros_per_1k: i64,
    pub request_micros: i64,
    pub ttft_bound_ms: i32,
    pub ttot_bound_ms: i32,
    pub active: bool,
    pub updated_at_cp: i64,
    pub updated_at_ms: i64,
}

/// Per-Settle denormalized row joining the voucher's cumulative_* with
/// the channel's snapshotted model_id + payee. One row per Settle tx;
/// INSERT-only (no upsert — the chain enforces tx_sequence_number is
/// unique).
#[derive(Insertable, Queryable, Debug, Clone, FieldCount)]
#[diesel(table_name = soma_inference_settlements)]
pub struct StoredInferenceSettlement {
    pub tx_sequence_number: i64,
    pub cp_sequence_number: i64,
    pub channel_id: Vec<u8>,
    pub payer: Vec<u8>,
    pub payee: Vec<u8>,
    pub model_id: String,
    pub cumulative_amount: i64,
    pub cumulative_prompt_tokens: i64,
    pub cumulative_completion_tokens: i64,
    pub cumulative_cache_read_tokens: i64,
    pub cumulative_cache_write_tokens: i64,
    pub cumulative_requests: i64,
    pub delta_amount: i64,
    pub timestamp_ms: i64,
}
