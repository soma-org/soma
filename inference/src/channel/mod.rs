// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-request authorization channel — the swap seam between the HTTP path
//! and the (eventually on-chain) settlement scheme.
//!
//! The MVP impl is [`running_tab::RunningTab`]: cumulative-authorization
//! Ed25519 signatures, one per request, monotonic per channel. The provider
//! could in principle present the latest signed cumulative on-chain to claim
//! that amount; today we settle off-chain and reconcile via batched
//! `PayProvider` transactions.

pub mod header;
pub mod running_tab;

use ::types::channel::Voucher;
use ::types::crypto::GenericSignature;
use async_trait::async_trait;

pub use running_tab::RunningTab;

#[derive(Debug, Clone)]
pub struct RequestMeta<'a> {
    pub method: &'a str,
    pub path: &'a str,
    pub body_sha256_hex: &'a str,
    pub timestamp_ms: u64,
    pub request_id: &'a str,
}

/// Per-request usage breakdown reported by an upstream OpenAI-compatible
/// provider. The relay extracts this from the `usage` block in the
/// streaming `[DONE]` chunk (or from the non-streaming JSON body) and
/// passes it to [`PaymentChannel::reconcile`] / `post_flight`, which
/// bump cumulative counters on the channel state. The on-chain voucher
/// the next `authorize` call signs carries the running totals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RequestUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
}

impl RequestUsage {
    /// Project an OpenAI-flavored `usage` block onto per-channel
    /// counters. Cached prompt tokens live in `prompt_tokens_details`
    /// (Anthropic-compatible providers populate that field) — we map
    /// `cached_tokens` onto `cache_read_tokens` and `cache_write_tokens`
    /// directly through.
    pub fn from_openai(u: &crate::openai::Usage) -> Self {
        let (cache_read, cache_write) = match &u.prompt_tokens_details {
            Some(d) => (d.cached_tokens, d.cache_write_tokens),
            None => (0, 0),
        };
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            cache_read_tokens: cache_read,
            cache_write_tokens: cache_write,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error("auth header missing or malformed")]
    Malformed,
    #[error("signature invalid")]
    BadSignature,
    #[error("expired")]
    Expired,
    #[error("non-monotonic cumulative")]
    NonMonotonic,
    #[error("over deposit")]
    OverDeposit,
    #[error("payment required: need {need_micros} authorized")]
    PaymentRequired { need_micros: u64 },
    #[error("not found")]
    NotFound,
    #[error("internal: {0}")]
    Internal(String),
}

#[async_trait]
pub trait PaymentChannel: Send + Sync + 'static {
    type ClientState: Send + Sync;
    type ProviderState: Send + Sync;

    async fn authorize(
        &self,
        state: &mut Self::ClientState,
        meta: &RequestMeta<'_>,
        worst_case_cost_micros: u64,
    ) -> Result<String, ChannelError>;

    async fn pre_flight(
        &self,
        state: &mut Self::ProviderState,
        header_value: &str,
        onchain_sig: &GenericSignature,
        meta: &RequestMeta<'_>,
        worst_case_cost_micros: u64,
    ) -> Result<(), ChannelError>;

    async fn post_flight(
        &self,
        state: &mut Self::ProviderState,
        meta: &RequestMeta<'_>,
        actual_cost_micros: u64,
        usage: RequestUsage,
    ) -> Result<(), ChannelError>;

    async fn reconcile(
        &self,
        state: &mut Self::ClientState,
        request_id: &str,
        actual_cost_micros: u64,
        usage: RequestUsage,
    );

    /// Returns the latest on-chain `Voucher` + signature held by the
    /// provider for this channel — exactly what `sdk::channel::settle`
    /// needs. `None` until the first request lands and the proxy
    /// supplies a sig.
    fn final_settlement(&self, state: &Self::ProviderState) -> Option<(Voucher, GenericSignature)>;
}
