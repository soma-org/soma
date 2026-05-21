// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cumulative-authorization "running tab" payment channel.
//!
//! Each request bumps the channel's cumulative authorized total and
//! the payer signs **two** vouchers per request via
//! `sdk::channel::{sign_http_voucher, sign_voucher}`:
//!
//!   - **HttpVoucher** (`IntentScope::PaymentVoucherHttp`) — bound to
//!     the HTTP request context (method, path, body, request id,
//!     expiry). Verified per request. Prevents an adversarial provider
//!     from replaying a signature against a different request.
//!   - **Voucher** (`IntentScope::PaymentVoucher`) — the on-chain
//!     pair `(channel_id, cumulative_amount)`. Stored by the provider
//!     and submitted via `sdk::channel::settle` to actually claim the
//!     funds.
//!
//! Both signatures are produced through the SDK's
//! `Signature::new_secure` path → same keystore, MultiSig support
//! transparent, single `GenericSignature` wire type.
//!
//! Constraints enforced by [`RunningTab::pre_flight`]:
//! - HTTP voucher signature valid under the channel's `authorized_signer`
//! - HttpVoucher binding fields match the actual request
//! - cumulative monotonic across requests
//! - cumulative ≤ deposit (over-deposit is rejected)
//! - cumulative ≥ already-consumed + worst-case (otherwise [`ChannelError::PaymentRequired`])

use std::collections::HashMap;
use std::sync::Arc;

use ::types::base::SomaAddress;
use ::types::channel::{Channel, HttpVoucher, Voucher};
use ::types::crypto::GenericSignature;
use ::types::object::ObjectID;
use async_trait::async_trait;
use sdk::wallet_context::WalletContext;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::channel::header::{SomaPayHeader, decode_onchain_sig, encode_onchain_sig};
use crate::channel::{ChannelError, PaymentChannel, RequestMeta, RequestUsage};
use crate::now_ms;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LastAuth {
    pub request_id: String,
    pub estimated_micros: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TabClientState {
    pub channel_id: ObjectID,
    pub provider_address: SomaAddress,
    pub provider_endpoint: String,
    pub deposit_micros: u64,
    pub cumulative_authorized_micros: u64,
    pub last_authorized: Option<LastAuth>,
    /// Map of request_id -> realized cost. Entry is consumed by the
    /// next `authorize` call that uses it.
    #[serde(default)]
    pub realized: HashMap<String, u64>,
    /// SLA bounds snapshotted onto the channel from the offering at
    /// open time. The relay layer compares measured TTFT/TTOT against
    /// these and emits negative `RateChannel` ratings on breach. `0`
    /// disables the check.
    #[serde(default)]
    pub ttft_bound_ms: u32,
    #[serde(default)]
    pub ttot_bound_ms: u32,
    /// Model id bound to this channel — set at slot install from the
    /// on-chain `Channel.model_id`. The relay rejects requests whose
    /// payload `model` doesn't match.
    #[serde(default)]
    pub model_id: String,
    /// Cumulative usage breakdown signed alongside `cumulative_authorized_micros`
    /// onto each on-chain voucher. The relay bumps these counters from the
    /// upstream `usage` block in `reconcile`, and the next `authorize` call
    /// embeds the running totals into the signed `Voucher` so the indexer
    /// can materialize exact per-channel token volume at Settle time.
    #[serde(default)]
    pub cumulative_prompt_tokens: u64,
    #[serde(default)]
    pub cumulative_completion_tokens: u64,
    #[serde(default)]
    pub cumulative_cache_read_tokens: u64,
    #[serde(default)]
    pub cumulative_cache_write_tokens: u64,
    #[serde(default)]
    pub cumulative_requests: u64,
    /// Last epoch in which the proxy emitted a `RateChannel(TtftBreach|TtotBreach)`
    /// for this channel. Used by the relay layer to rate-limit breach ratings
    /// to at most one per channel per epoch (a tx per breach would balloon gas
    /// without adding observability — the indexer aggregates ratings per epoch).
    /// `0` means "never emitted"; the next breach can fire freely.
    #[serde(default)]
    pub last_breach_emitted_at_epoch: u64,
}

impl TabClientState {
    pub fn new(
        channel_id: ObjectID,
        provider_address: SomaAddress,
        provider_endpoint: String,
        deposit_micros: u64,
    ) -> Self {
        Self {
            channel_id,
            provider_address,
            provider_endpoint,
            deposit_micros,
            cumulative_authorized_micros: 0,
            last_authorized: None,
            realized: HashMap::new(),
            ttft_bound_ms: 0,
            ttot_bound_ms: 0,
            model_id: String::new(),
            cumulative_prompt_tokens: 0,
            cumulative_completion_tokens: 0,
            cumulative_cache_read_tokens: 0,
            cumulative_cache_write_tokens: 0,
            cumulative_requests: 0,
            last_breach_emitted_at_epoch: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TabProviderState {
    pub channel_id: ObjectID,
    /// Address whose key signs vouchers. Snapshot of
    /// `Channel.authorized_signer` at slot init — used to verify the
    /// HTTP voucher signature. The channel's `authorized_signer`
    /// can't be mutated on-chain, so this never goes stale.
    pub authorized_signer: SomaAddress,
    pub deposit_micros: u64,
    pub cumulative_authorized_micros: u64,
    pub total_consumed_micros: u64,
    pub last_request_id: Option<String>,
    /// Latest on-chain `Voucher` signature received for this channel.
    /// `Settle` carries this. Empty until the first request lands.
    #[serde(default, with = "onchain_sig_serde")]
    pub last_onchain_sig: Option<GenericSignature>,
    /// Cumulative amount that has actually been settled on-chain so
    /// far — i.e., the highest `cumulative_amount` we've successfully
    /// submitted via `Settle`. The background ticker compares
    /// `cumulative_authorized_micros` against this value to skip
    /// channels with no progress, avoiding wasteful gas. Initialized
    /// to the on-chain `Channel.settled_amount` at slot install so a
    /// freshly-rehydrated slot doesn't double-settle.
    ///
    /// `#[serde(default)]` lets older persisted ledgers (without this
    /// field) deserialize: 0 is a safe default — the ticker just sees
    /// "no progress recorded" and waits for the next live update.
    #[serde(default)]
    pub last_settled_at_amount: u64,
    /// Cumulative usage breakdown tracked from upstream `post_flight` calls.
    /// Persisted into `final_settlement`'s `Voucher` so the on-chain
    /// signature validates against the same per-channel totals the indexer
    /// records.
    #[serde(default)]
    pub cumulative_prompt_tokens: u64,
    #[serde(default)]
    pub cumulative_completion_tokens: u64,
    #[serde(default)]
    pub cumulative_cache_read_tokens: u64,
    #[serde(default)]
    pub cumulative_cache_write_tokens: u64,
    #[serde(default)]
    pub cumulative_requests: u64,
    /// Snapshot of the Voucher fields the *client* signed when it issued
    /// the most recent on-chain signature in `last_onchain_sig`. The
    /// client signs over `(channel_id, cumulative_amount, prompt_tokens,
    /// completion_tokens, cache_read_tokens, cache_write_tokens, requests)`
    /// using its own counters at sign time (pre-reconcile). The provider
    /// must reconstruct exactly that voucher at settle time — otherwise
    /// the chain-side signature verification fails. The provider's own
    /// `cumulative_*_tokens` counters are advanced by `post_flight` and
    /// therefore drift past the signed values for the most recent
    /// request, so we can't reuse them as-is.
    #[serde(default)]
    pub last_signed_cumulative_amount: u64,
    #[serde(default)]
    pub last_signed_prompt_tokens: u64,
    #[serde(default)]
    pub last_signed_completion_tokens: u64,
    #[serde(default)]
    pub last_signed_cache_read_tokens: u64,
    #[serde(default)]
    pub last_signed_cache_write_tokens: u64,
    #[serde(default)]
    pub last_signed_requests: u64,
}

mod onchain_sig_serde {
    use ::types::crypto::GenericSignature;
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use fastcrypto::traits::ToFromBytes as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<GenericSignature>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            None => Option::<String>::None.serialize(s),
            Some(sig) => Some(URL_SAFE_NO_PAD.encode(sig.as_ref())).serialize(s),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<GenericSignature>, D::Error> {
        let opt = Option::<String>::deserialize(d)?;
        match opt {
            None => Ok(None),
            Some(s) => {
                let bytes =
                    URL_SAFE_NO_PAD.decode(s.as_bytes()).map_err(serde::de::Error::custom)?;
                let sig = GenericSignature::from_bytes(&bytes).map_err(serde::de::Error::custom)?;
                Ok(Some(sig))
            }
        }
    }
}

impl TabProviderState {
    pub fn new(channel_id: ObjectID, chan: &Channel) -> Self {
        let on_chain_settled = chan.settled_amount();
        Self {
            channel_id,
            authorized_signer: chan.authorized_signer(),
            deposit_micros: chan.deposit(),
            cumulative_authorized_micros: on_chain_settled,
            total_consumed_micros: on_chain_settled,
            last_request_id: None,
            last_onchain_sig: None,
            // Match the chain's view — anything below this would
            // be rejected by the executor as non-monotonic anyway.
            last_settled_at_amount: on_chain_settled,
            cumulative_prompt_tokens: 0,
            cumulative_completion_tokens: 0,
            cumulative_cache_read_tokens: 0,
            cumulative_cache_write_tokens: 0,
            cumulative_requests: 0,
            last_signed_cumulative_amount: on_chain_settled,
            last_signed_prompt_tokens: 0,
            last_signed_completion_tokens: 0,
            last_signed_cache_read_tokens: 0,
            last_signed_cache_write_tokens: 0,
            last_signed_requests: 0,
        }
    }
}

/// Auth validity window for HTTP vouchers (per request expiry).
const AUTH_VALIDITY_SECS: u64 = 60;

pub struct RunningTab {
    /// Set on the client side (so it can sign); absent on the
    /// provider side. The signer address used inside the WalletContext.
    signing: Option<(Arc<WalletContext>, SomaAddress)>,
    pub clock_skew_tolerance_secs: u64,
    pub auth_validity_secs: u64,
}

impl RunningTab {
    pub fn for_provider(clock_skew_tolerance_secs: u64) -> Self {
        Self { signing: None, clock_skew_tolerance_secs, auth_validity_secs: AUTH_VALIDITY_SECS }
    }

    pub fn for_client(ctx: Arc<WalletContext>, signer: SomaAddress) -> Self {
        Self {
            signing: Some((ctx, signer)),
            clock_skew_tolerance_secs: 60,
            auth_validity_secs: AUTH_VALIDITY_SECS,
        }
    }

    fn body_sha256_from_hex(hex_str: &str) -> Result<[u8; 32], ChannelError> {
        let bytes = hex::decode(hex_str).map_err(|_| ChannelError::Malformed)?;
        if bytes.len() != 32 {
            return Err(ChannelError::Malformed);
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(&bytes);
        Ok(out)
    }

    fn request_id_sha(rid: &str) -> [u8; 32] {
        Sha256::digest(rid.as_bytes()).into()
    }

    fn method_path_sha(method: &str, path: &str) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(method.as_bytes());
        h.update(b"\n");
        h.update(path.as_bytes());
        h.finalize().into()
    }
}

#[async_trait]
impl PaymentChannel for RunningTab {
    type ClientState = TabClientState;
    type ProviderState = TabProviderState;

    async fn authorize(
        &self,
        state: &mut Self::ClientState,
        meta: &RequestMeta<'_>,
        worst_case_cost_micros: u64,
    ) -> Result<String, ChannelError> {
        let (ctx, signer) = self
            .signing
            .as_ref()
            .ok_or_else(|| ChannelError::Internal("no signing context".into()))?;

        // 1. Slack from previous request, if any.
        let slack = match &state.last_authorized {
            Some(la) => state
                .realized
                .get(&la.request_id)
                .copied()
                .map(|r| la.estimated_micros.saturating_sub(r))
                .unwrap_or(0),
            None => 0,
        };

        // 2. Bump (≥1µ so cumulative always advances).
        let bump_raw = worst_case_cost_micros.saturating_sub(slack);
        let bump = bump_raw.max(1);
        let new_cum = state.cumulative_authorized_micros.saturating_add(bump);
        if new_cum > state.deposit_micros {
            return Err(ChannelError::Internal(
                "would exceed deposit; soma channel top-up needed".into(),
            ));
        }
        let expires_ms = now_ms() + self.auth_validity_secs * 1000;

        // 3. Build the on-chain Voucher (cumulative_amount + per-channel
        //    usage breakdown). The HttpVoucher will embed this verbatim
        //    so the provider stores exactly what we sign — no separate
        //    re-derivation that has to byte-match our view across every
        //    edge case (multiple SSE `usage` events, missed reconciles,
        //    upstream parsing differences). This makes the price/token
        //    oracle on chain trustworthy without making settle correctness
        //    depend on byte-perfect agreement on counters.
        let voucher = Voucher::new(
            state.channel_id,
            new_cum,
            state.cumulative_prompt_tokens,
            state.cumulative_completion_tokens,
            state.cumulative_cache_read_tokens,
            state.cumulative_cache_write_tokens,
            state.cumulative_requests,
        );

        // 4. Sign the on-chain Voucher. The provider will pair this with
        //    the embedded voucher at settle time; the chain verifies sig
        //    against the embedded voucher and that's it.
        let onchain_sig = sdk::channel::sign_voucher(&ctx.config.keystore, signer, &voucher)
            .await
            .map_err(|e| ChannelError::Internal(format!("sign_voucher: {e}")))?;

        // 5. Build the HttpVoucher embedding the on-chain voucher, sign it
        //    for the HTTP layer. The HTTP sig additionally binds the
        //    per-request HTTP context (body/method/path/expiry) so the
        //    provider can't replay this signature against a different
        //    request.
        let body_sha256 = Self::body_sha256_from_hex(meta.body_sha256_hex)?;
        let request_id_sha = Self::request_id_sha(meta.request_id);
        let method_path_sha = Self::method_path_sha(meta.method, meta.path);
        let http_voucher =
            HttpVoucher::new(voucher, expires_ms, body_sha256, request_id_sha, method_path_sha);
        let http_sig = sdk::channel::sign_http_voucher(&ctx.config.keystore, signer, &http_voucher)
            .await
            .map_err(|e| ChannelError::Internal(format!("sign_http_voucher: {e}")))?;

        // 5. Commit local state.
        state.cumulative_authorized_micros = new_cum;
        if let Some(la) = state.last_authorized.as_ref() {
            state.realized.remove(&la.request_id);
        }
        state.last_authorized = Some(LastAuth {
            request_id: meta.request_id.to_string(),
            estimated_micros: worst_case_cost_micros,
        });

        let header = SomaPayHeader { channel_id: state.channel_id, http_voucher, http_sig };
        // We pack the on-chain sig into the same string return so
        // callers don't need a second function — proxy/relay parses
        // both pieces and ships them as separate HTTP headers. The
        // delimiter `||` cannot appear in URL_SAFE_NO_PAD base64.
        Ok(format!("{}||{}", header.format(), encode_onchain_sig(&onchain_sig)))
    }

    async fn pre_flight(
        &self,
        state: &mut Self::ProviderState,
        header_value: &str,
        onchain_sig: &GenericSignature,
        meta: &RequestMeta<'_>,
        worst_case_cost_micros: u64,
    ) -> Result<(), ChannelError> {
        let header = SomaPayHeader::parse(header_value)?;

        // Expiry (with clock-skew tolerance).
        let now = now_ms();
        let tol_ms = self.clock_skew_tolerance_secs * 1000;
        if header.http_voucher.expires_ms() + tol_ms < now {
            return Err(ChannelError::Expired);
        }
        if header.channel_id != state.channel_id {
            return Err(ChannelError::NotFound);
        }

        // HttpVoucher binding fields must match the actual request.
        let body_sha256 = Self::body_sha256_from_hex(meta.body_sha256_hex)?;
        if header.http_voucher.body_sha256() != body_sha256 {
            return Err(ChannelError::BadSignature);
        }
        if header.http_voucher.request_id_sha256() != Self::request_id_sha(meta.request_id) {
            return Err(ChannelError::BadSignature);
        }
        if header.http_voucher.method_path_sha256() != Self::method_path_sha(meta.method, meta.path)
        {
            return Err(ChannelError::BadSignature);
        }

        // Signature pair: both HTTP-layer and on-chain layer signatures
        // must verify against the channel's `authorized_signer` at
        // receive time. The on-chain sig verifies against the embedded
        // `Voucher` — the same bytes we'll submit at settle — so if it
        // verifies here, settle is guaranteed to verify too. Fail-fast
        // catches a corrupted/tampered/wrong-key voucher at the first
        // request, not hours later at shutdown.
        let channel = synthetic_channel(state.authorized_signer);
        sdk::channel::verify_http_voucher(&channel, &header.http_voucher, &header.http_sig)
            .map_err(|_| ChannelError::BadSignature)?;
        sdk::channel::verify_voucher(&channel, *header.http_voucher.voucher(), onchain_sig)
            .map_err(|_| ChannelError::BadSignature)?;

        // Monotonic: same request id may re-present same cum;
        // otherwise must strictly increase.
        let voucher = header.http_voucher.voucher();
        let same_request =
            state.last_request_id.as_deref().map(|r| r == meta.request_id).unwrap_or(false);
        let cum = voucher.cumulative_amount();
        if cum < state.cumulative_authorized_micros {
            return Err(ChannelError::NonMonotonic);
        }
        if cum == state.cumulative_authorized_micros && !same_request {
            return Err(ChannelError::NonMonotonic);
        }
        // Deposit cap.
        if cum > state.deposit_micros {
            return Err(ChannelError::OverDeposit);
        }
        // Payment required: the new authorization must cover already-consumed + worst-case.
        let need = state.total_consumed_micros.saturating_add(worst_case_cost_micros);
        if cum < need {
            return Err(ChannelError::PaymentRequired { need_micros: need });
        }

        // Copy the voucher's fields verbatim into `last_signed_*`. The
        // payer signed *these exact bytes*, so reusing them at settle
        // means the chain verifies by construction. Old design re-derived
        // them from the provider's own usage tracking, which had to
        // byte-match the payer's view across every edge case (multiple
        // SSE `usage` events, missed reconciles, parsing differences) —
        // any drift silently broke settle hours later.
        state.last_signed_cumulative_amount = cum;
        state.last_signed_prompt_tokens = voucher.cumulative_prompt_tokens();
        state.last_signed_completion_tokens = voucher.cumulative_completion_tokens();
        state.last_signed_cache_read_tokens = voucher.cumulative_cache_read_tokens();
        state.last_signed_cache_write_tokens = voucher.cumulative_cache_write_tokens();
        state.last_signed_requests = voucher.cumulative_requests();

        state.cumulative_authorized_micros = cum;
        state.last_request_id = Some(meta.request_id.to_string());
        Ok(())
    }

    async fn post_flight(
        &self,
        state: &mut Self::ProviderState,
        _meta: &RequestMeta<'_>,
        actual_cost_micros: u64,
        usage: RequestUsage,
    ) -> Result<(), ChannelError> {
        state.total_consumed_micros =
            state.total_consumed_micros.saturating_add(actual_cost_micros);
        state.cumulative_prompt_tokens =
            state.cumulative_prompt_tokens.saturating_add(usage.prompt_tokens);
        state.cumulative_completion_tokens =
            state.cumulative_completion_tokens.saturating_add(usage.completion_tokens);
        state.cumulative_cache_read_tokens =
            state.cumulative_cache_read_tokens.saturating_add(usage.cache_read_tokens);
        state.cumulative_cache_write_tokens =
            state.cumulative_cache_write_tokens.saturating_add(usage.cache_write_tokens);
        state.cumulative_requests = state.cumulative_requests.saturating_add(1);
        if state.total_consumed_micros > state.cumulative_authorized_micros {
            tracing::warn!(
                channel = %state.channel_id,
                consumed = state.total_consumed_micros,
                authorized = state.cumulative_authorized_micros,
                "consumed exceeds authorized; next pre_flight will reject"
            );
        }
        Ok(())
    }

    async fn reconcile(
        &self,
        state: &mut Self::ClientState,
        request_id: &str,
        actual_cost_micros: u64,
        usage: RequestUsage,
    ) {
        state.realized.insert(request_id.to_string(), actual_cost_micros);
        state.cumulative_prompt_tokens =
            state.cumulative_prompt_tokens.saturating_add(usage.prompt_tokens);
        state.cumulative_completion_tokens =
            state.cumulative_completion_tokens.saturating_add(usage.completion_tokens);
        state.cumulative_cache_read_tokens =
            state.cumulative_cache_read_tokens.saturating_add(usage.cache_read_tokens);
        state.cumulative_cache_write_tokens =
            state.cumulative_cache_write_tokens.saturating_add(usage.cache_write_tokens);
        state.cumulative_requests = state.cumulative_requests.saturating_add(1);
    }

    fn final_settlement(&self, state: &Self::ProviderState) -> Option<(Voucher, GenericSignature)> {
        state.last_onchain_sig.as_ref().map(|sig| {
            // Settlement voucher MUST be rebuilt from the exact fields the
            // client signed over for the most recent on-chain voucher.
            // Provider-side `cumulative_*_tokens` (bumped by `post_flight`)
            // drift past the signed values by the most recent request's
            // usage, so they would not match the signature. Use the
            // snapshot captured in `pre_flight` instead.
            (
                Voucher::new(
                    state.channel_id,
                    state.last_signed_cumulative_amount,
                    state.last_signed_prompt_tokens,
                    state.last_signed_completion_tokens,
                    state.last_signed_cache_read_tokens,
                    state.last_signed_cache_write_tokens,
                    state.last_signed_requests,
                ),
                sig.clone(),
            )
        })
    }
}

/// Build a stand-in `Channel` for signature verification — only
/// `authorized_signer` is read by `verify_http_voucher`. Avoids a
/// fresh chain read on every request.
fn synthetic_channel(authorized_signer: SomaAddress) -> Channel {
    Channel::new_for_testing(
        SomaAddress::ZERO,
        SomaAddress::ZERO,
        authorized_signer,
        ::types::object::CoinType::Usdc,
        0,
    )
}

/// Helper used by the auth middleware to split the combined header
/// value (`<somapay_header>||<onchain_sig_b64>`) into the two pieces
/// the provider needs.
pub fn split_combined_header(value: &str) -> Result<(String, GenericSignature), ChannelError> {
    let mut parts = value.splitn(2, "||");
    let header = parts.next().ok_or(ChannelError::Malformed)?.to_string();
    let onchain_b64 = parts.next().ok_or(ChannelError::Malformed)?;
    let sig = decode_onchain_sig(onchain_b64)?;
    Ok((header, sig))
}
