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
        meta: &RequestMeta<'_>,
        worst_case_cost_micros: u64,
    ) -> Result<(), ChannelError>;

    async fn post_flight(
        &self,
        state: &mut Self::ProviderState,
        meta: &RequestMeta<'_>,
        actual_cost_micros: u64,
    ) -> Result<(), ChannelError>;

    async fn reconcile(
        &self,
        state: &mut Self::ClientState,
        request_id: &str,
        actual_cost_micros: u64,
    );

    fn final_settlement(&self, state: &Self::ProviderState) -> (u64, String);
}
