// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Voucher-side token usage projections.
//!
//! Each `Settle` tx carries the running per-channel cumulative_*
//! totals signed by the buyer. The `soma_inference_settlements` table
//! materializes one row per `Settle` tx (denormalized with the
//! channel's `model_id` and `payee`). Cumulative totals are monotonic
//! non-decreasing per channel, so the *latest* row for a channel is
//! that channel's lifetime usage.
//!
//! Two surfaces are exposed:
//!   * `ChannelUsage` — per-channel lifetime totals (Channel.usage)
//!   * `TokenUsageTotals` — aggregated by (payer | model_id) for the
//!     leaderboard / profile pages

use async_graphql::*;

use crate::api::scalars::{BigInt, SomaAddress};

/// Latest cumulative voucher totals for one channel — i.e., this
/// channel's lifetime token usage and amount as of the most recent
/// `Settle` tx that touched it. `None`-valued channel-level field
/// when the channel has never been settled (no `Settle` rows in
/// `soma_inference_settlements` for this `channel_id`).
#[derive(Clone, Debug, SimpleObject)]
pub struct ChannelUsage {
    /// Cumulative voucher amount paid out (micro-USDC). Equal to
    /// `Channel.settledAmount` at the same checkpoint.
    pub cumulative_amount: BigInt,
    pub cumulative_prompt_tokens: BigInt,
    pub cumulative_completion_tokens: BigInt,
    pub cumulative_cache_read_tokens: BigInt,
    pub cumulative_cache_write_tokens: BigInt,
    pub cumulative_requests: BigInt,
    /// Timestamp of the most recent `Settle` tx on this channel.
    pub last_settlement_ms: BigInt,
}

/// Aggregated token usage and spend for one key — either a `payer`
/// address (user leaderboard / profile) or a `model_id` (model
/// leaderboard / model detail). Summed across the latest cumulative
/// of every channel that matches the key, so re-Settles on the same
/// channel are *not* double-counted.
#[derive(Clone, Debug, SimpleObject)]
pub struct TokenUsageTotals {
    /// Hex address (payer leaderboard) or model identifier (model
    /// leaderboard). The query that produced this row knows which.
    pub key: String,
    /// `key` as a hex address when this row is keyed by payer;
    /// `None` when keyed by `model_id`. Lets clients render the key
    /// as an address link without re-parsing.
    pub payer: Option<SomaAddress>,
    /// `key` as a model identifier when this row is keyed by model;
    /// `None` when keyed by payer.
    pub model_id: Option<String>,
    /// Sum of channels' latest `cumulative_amount` (micro-USDC).
    pub total_amount: BigInt,
    pub total_prompt_tokens: BigInt,
    pub total_completion_tokens: BigInt,
    pub total_cache_read_tokens: BigInt,
    pub total_cache_write_tokens: BigInt,
    pub total_requests: BigInt,
    /// Number of distinct channels contributing to the totals.
    pub channel_count: i32,
}
