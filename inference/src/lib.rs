// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Inference proxy + provider server.
//!
//! Two halves of an OpenAI-compatible inference channel:
//! - [`proxy`] — local agent-facing proxy. The agent CLI points at it via
//!   `OPENAI_BASE_URL`. The proxy picks a provider for the requested model,
//!   opens (or reuses) a payment channel, signs each request with a
//!   [`channel::PaymentChannel`], relays SSE through, and reconciles realized
//!   cost back into the channel state.
//! - [`server`] — provider-side server fronting an OpenAI-compatible upstream
//!   (OpenRouter, Vast vLLM, …). Verifies the per-request authorization,
//!   holds the per-channel mutex through the upstream call, runs post-flight
//!   on the realized usage.
//!
//! Two swap seams:
//! - [`chain::Discovery`] — provider registry (filesystem MVP, on-chain later).
//! - [`channel::PaymentChannel`] — per-request authorization scheme.
//!   [`channel::RunningTab`] is the cumulative-authorization MVP impl.

pub mod catalog;
pub mod chain;
pub mod channel;
pub mod openai;
pub mod persist;
pub mod pricing;
pub mod tokenizer;
pub mod proxy;
pub mod server;

pub mod http_util;

pub use chain::{Discovery, ProviderRecord, ChannelHandle, ChannelState, ChannelStatus, OpenChannelParams, ChainError};
pub use channel::{PaymentChannel, RequestMeta, ChannelError, RunningTab};

/// UNIX time in milliseconds.
pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Fresh request id for tracing the wire.
pub fn new_request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}
