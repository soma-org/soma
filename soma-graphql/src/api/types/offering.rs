// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use async_graphql::*;

use crate::api::scalars::{BigInt, SomaAddress};

/// Off-chain projection of an on-chain `(provider, model_id)`
/// `Offering` shared object — sourced from the `soma_offerings`
/// indexer table.
///
/// The proxy uses this to discover what models a provider serves and
/// at what price, replacing the per-provider HTTP `/v1/models`
/// fan-out: the chain (and indexer mirror) is authoritative, so a
/// single GraphQL query gives every active offering across every
/// provider.
#[derive(Clone, Debug)]
pub struct Offering {
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

#[Object]
impl Offering {
    /// Provider address — the payee on any channel opened against
    /// this offering.
    async fn provider(&self) -> SomaAddress {
        SomaAddress(self.provider.clone())
    }

    /// Canonical model id from the protocol-config `ModelRegistry`
    /// (e.g. `anthropic/claude-haiku-4.5`).
    async fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Price for input/prompt tokens, in micro-USD per 1000 tokens.
    /// (1 micro = $1e-6, so `1000` here = $0.001/1k = $1/1M tokens.)
    async fn prompt_micros_per_1k(&self) -> BigInt {
        BigInt(self.prompt_micros_per_1k)
    }

    /// Price for output/completion tokens, in micro-USD per 1000 tokens.
    async fn completion_micros_per_1k(&self) -> BigInt {
        BigInt(self.completion_micros_per_1k)
    }

    /// Price for cache-read prompt tokens, in micro-USD per 1000 tokens.
    async fn cache_read_micros_per_1k(&self) -> BigInt {
        BigInt(self.cache_read_micros_per_1k)
    }

    /// Price for cache-write prompt tokens, in micro-USD per 1000 tokens.
    async fn cache_write_micros_per_1k(&self) -> BigInt {
        BigInt(self.cache_write_micros_per_1k)
    }

    /// Flat per-request surcharge in micro-USD.
    async fn request_micros(&self) -> BigInt {
        BigInt(self.request_micros)
    }

    /// Time-to-first-token SLA bound (ms). `0` disables the check.
    /// Channels opened against this offering snapshot this value.
    async fn ttft_bound_ms(&self) -> i32 {
        self.ttft_bound_ms
    }

    /// Time-to-output-token SLA bound (ms per output token). `0`
    /// disables.
    async fn ttot_bound_ms(&self) -> i32 {
        self.ttot_bound_ms
    }

    /// `false` once `DeactivateOffering` lands. Channels opened
    /// against an inactive offering are rejected with
    /// `ChannelOfferingMissing`.
    async fn active(&self) -> bool {
        self.active
    }

    /// Last checkpoint at which this offering's row changed
    /// (register / update / deactivate). Use as a freshness cursor.
    async fn updated_at_cp(&self) -> BigInt {
        BigInt(self.updated_at_cp)
    }

    /// Wall-clock ms at which this offering's row changed. Mirrors
    /// the on-chain `Offering.updated_at_ms`.
    async fn updated_at_ms(&self) -> BigInt {
        BigInt(self.updated_at_ms)
    }
}
