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
        model_id: String,
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
            model_id,
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
    /// Model id bound to this channel on-chain. Snapshotted from
    /// `Channel.model_id` at slot install. The relay rejects any
    /// chat request whose payload `model` differs from this value —
    /// channels are scoped per `(payer, provider, model_id)` and
    /// pricing/usage accounting is per-channel.
    pub model_id: String,
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
            model_id: chan.model_id().to_string(),
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

        // Channel is bound on-chain to a single `model_id` (snapshotted
        // at `OpenChannel` time). Reject requests that target a different
        // model — they belong on a different channel; reusing this one
        // would break the per-(payer, payee, model_id) accounting.
        if meta.model_id != state.model_id {
            return Err(ChannelError::ModelMismatch {
                expected: state.model_id.clone(),
                got: meta.model_id.to_string(),
            });
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

        // Monotonic + payment-required check. A voucher whose
        // `cumulative_amount()` is below what this provider has already
        // authorized for the channel is *not* a replay attack — the
        // signature, body sha, request_id, and method/path were all
        // verified above, so the payer signed *this exact request*. The
        // realistic cause is a payer-side proxy restart where the proxy
        // rehydrated state from the indexer (`settled_amount`), which lags
        // the provider's in-memory `cumulative_authorized_micros` between
        // auto-settle ticks. Return `PaymentRequired` with `need_micros`
        // pointing at the floor so the proxy's `relay.rs:151` retry path
        // bumps and re-signs — instead of `auth_invalid`, which the proxy
        // has no recovery path for and which surfaces as a hard 401 to the
        // user.
        //
        // The `total_consumed + worst_case` floor still applies; we take
        // the max so a stale-rehydrate proxy is told the *higher* of "you
        // owe me what I've already authorized" and "you need to cover
        // worst case for this request."
        let voucher = header.http_voucher.voucher();
        let same_request =
            state.last_request_id.as_deref().map(|r| r == meta.request_id).unwrap_or(false);
        let cum = voucher.cumulative_amount();
        // Deposit cap is the one genuinely-unrecoverable case: the
        // channel can never authorize beyond its deposit, so a voucher
        // asking for more is malformed. The proxy's retry path can't fix
        // this — only a `TopUp` (or fresh channel) can.
        if cum > state.deposit_micros {
            return Err(ChannelError::OverDeposit);
        }
        let monotonic_floor = if same_request {
            state.cumulative_authorized_micros
        } else {
            state.cumulative_authorized_micros.saturating_add(1)
        };
        let need = state
            .total_consumed_micros
            .saturating_add(worst_case_cost_micros)
            .max(monotonic_floor);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::PaymentChannel;
    use soma_keys::keystore::{AccountKeystore, InMemKeystore, Keystore};
    use std::sync::Arc;

    /// Construct a (RunningTab client, signer address) backed by an
    /// in-memory keystore so the test can sign vouchers without touching
    /// a filesystem keystore.
    fn ephemeral_client() -> (RunningTab, SomaAddress, Arc<WalletContext>) {
        let keystore = Keystore::InMem(InMemKeystore::new_insecure_for_tests(1));
        let signer = *keystore.addresses().first().expect("one key");
        let ctx = Arc::new(WalletContext::new_for_tests(keystore, None, None));
        (RunningTab::for_client(ctx.clone(), signer), signer, ctx)
    }

    fn make_client_state(
        channel_id: ObjectID,
        signer: SomaAddress,
        cumulative_authorized_micros: u64,
        deposit_micros: u64,
    ) -> TabClientState {
        let mut s = TabClientState::new(
            channel_id,
            signer,
            "http://example.invalid".to_string(),
            "anthropic/claude-haiku-4.5".to_string(),
            deposit_micros,
        );
        s.cumulative_authorized_micros = cumulative_authorized_micros;
        s
    }

    fn make_provider_state(
        channel_id: ObjectID,
        signer: SomaAddress,
        cumulative_authorized_micros: u64,
        deposit_micros: u64,
    ) -> TabProviderState {
        TabProviderState {
            channel_id,
            authorized_signer: signer,
            deposit_micros,
            cumulative_authorized_micros,
            total_consumed_micros: cumulative_authorized_micros,
            last_request_id: None,
            last_onchain_sig: None,
            last_settled_at_amount: cumulative_authorized_micros,
            cumulative_prompt_tokens: 0,
            cumulative_completion_tokens: 0,
            cumulative_cache_read_tokens: 0,
            cumulative_cache_write_tokens: 0,
            cumulative_requests: 0,
            last_signed_cumulative_amount: cumulative_authorized_micros,
            last_signed_prompt_tokens: 0,
            last_signed_completion_tokens: 0,
            last_signed_cache_read_tokens: 0,
            last_signed_cache_write_tokens: 0,
            last_signed_requests: 0,
            model_id: "anthropic/claude-haiku-4.5".to_string(),
        }
    }

    /// Reproduces the production bug we hit on testnet: the payer-side
    /// proxy rehydrates `cumulative_authorized_micros` from the indexer's
    /// stale `settled_amount`, then signs a voucher whose `cum` is below
    /// the provider's in-memory floor. Before the fix this returned
    /// `NonMonotonic` → relay surfaces it as `401 auth_invalid` → no
    /// retry path → user sees a hard failure. After the fix the provider
    /// returns `PaymentRequired { need_micros }` so the proxy's `relay.rs`
    /// retry path can resync.
    #[tokio::test]
    async fn stale_proxy_rehydrate_returns_payment_required_not_non_monotonic() {
        let (tab, signer, _ctx) = ephemeral_client();
        let channel_id = ObjectID::random();
        let body = b"{\"model\":\"anthropic/claude-haiku-4.5\"}";
        let body_sha = Sha256::digest(body);
        let body_sha_hex = hex::encode(body_sha);
        let request_id = "req-stale-rehydrate";
        let meta = RequestMeta {
            method: "POST",
            path: "/v1/chat/completions",
            body_sha256_hex: &body_sha_hex,
            timestamp_ms: now_ms(),
            request_id,
            model_id: "anthropic/claude-haiku-4.5",
        };
        let worst_case = 500u64;
        let deposit = 5_000_000u64;

        // Client (proxy after restart) thinks the channel is at
        // settled=10_000 — what the indexer reported.
        let mut client_state = make_client_state(channel_id, signer, 10_000, deposit);
        let combined = tab.authorize(&mut client_state, &meta, worst_case).await.expect("sign ok");
        let (header_str, onchain_sig) = split_combined_header(&combined).expect("split ok");

        // Provider's in-memory state is ahead of indexer — it has
        // authorized up to 200_000 already and hasn't called Settle yet.
        let mut provider_state = make_provider_state(channel_id, signer, 200_000, deposit);

        let result = tab
            .pre_flight(&mut provider_state, &header_str, &onchain_sig, &meta, worst_case)
            .await;

        match result {
            Err(ChannelError::PaymentRequired { need_micros }) => {
                assert!(
                    need_micros >= 200_001,
                    "need_micros should signal the provider's floor (got {need_micros})",
                );
            }
            Err(ChannelError::NonMonotonic) => {
                panic!(
                    "regression: stale-rehydrate returned NonMonotonic — \
                     the proxy's relay.rs:151 only retries on 402 PaymentRequired"
                );
            }
            other => panic!("expected PaymentRequired, got {other:?}"),
        }
    }

    /// Happy path: client and provider agree on the floor; first request
    /// after a fresh open succeeds.
    #[tokio::test]
    async fn fresh_channel_authorize_then_pre_flight_succeeds() {
        let (tab, signer, _ctx) = ephemeral_client();
        let channel_id = ObjectID::random();
        let body = b"{\"model\":\"anthropic/claude-haiku-4.5\"}";
        let body_sha = Sha256::digest(body);
        let body_sha_hex = hex::encode(body_sha);
        let meta = RequestMeta {
            method: "POST",
            path: "/v1/chat/completions",
            body_sha256_hex: &body_sha_hex,
            timestamp_ms: now_ms(),
            request_id: "req-fresh-1",
            model_id: "anthropic/claude-haiku-4.5",
        };
        let worst_case = 500u64;
        let deposit = 5_000_000u64;

        let mut client_state = make_client_state(channel_id, signer, 0, deposit);
        let combined = tab.authorize(&mut client_state, &meta, worst_case).await.expect("sign ok");
        let (header_str, onchain_sig) = split_combined_header(&combined).expect("split ok");

        let mut provider_state = make_provider_state(channel_id, signer, 0, deposit);
        tab.pre_flight(&mut provider_state, &header_str, &onchain_sig, &meta, worst_case)
            .await
            .expect("first request after fresh open should pre_flight cleanly");
    }

    /// End-to-end integration of the production fix: mirrors what
    /// `relay.rs:151` does when the provider returns 402 after the proxy
    /// rehydrates stale state. Validates that, after a single bump, the
    /// re-signed voucher passes `pre_flight` — i.e., the user's chat
    /// succeeds transparently instead of failing with `auth_invalid`.
    #[tokio::test]
    async fn stale_proxy_then_402_retry_succeeds_end_to_end() {
        let (tab, signer, _ctx) = ephemeral_client();
        let channel_id = ObjectID::random();
        let body = b"{\"model\":\"anthropic/claude-haiku-4.5\"}";
        let body_sha_hex = hex::encode(Sha256::digest(body));
        let request_id = "req-e2e-retry";
        let meta = RequestMeta {
            method: "POST",
            path: "/v1/chat/completions",
            body_sha256_hex: &body_sha_hex,
            timestamp_ms: now_ms(),
            request_id,
            model_id: "anthropic/claude-haiku-4.5",
        };
        let worst_case = 500u64;
        let deposit = 5_000_000u64;

        // Proxy rehydrates from indexer's stale settled_amount.
        let mut client_state = make_client_state(channel_id, signer, 10_000, deposit);
        // Provider has authorized further in memory.
        let mut provider_state = make_provider_state(channel_id, signer, 200_000, deposit);

        // === ROUND 1: stale voucher ===
        let combined1 = tab.authorize(&mut client_state, &meta, worst_case).await.unwrap();
        let (header1, sig1) = split_combined_header(&combined1).unwrap();
        let need = match tab.pre_flight(&mut provider_state, &header1, &sig1, &meta, worst_case).await
        {
            Err(ChannelError::PaymentRequired { need_micros }) => need_micros,
            other => panic!("expected PaymentRequired on stale voucher, got {other:?}"),
        };
        assert!(need >= 200_001, "need_micros must signal the provider's floor");

        // === ROUND 2: relay.rs:151 resync — set cum_authorized to need - worst_case,
        // clear last_authorized, re-call authorize. This is exactly what production
        // does on 402. ===
        client_state.cumulative_authorized_micros = need.saturating_sub(worst_case);
        client_state.last_authorized = None;
        let combined2 = tab.authorize(&mut client_state, &meta, worst_case).await.unwrap();
        let (header2, sig2) = split_combined_header(&combined2).unwrap();
        tab.pre_flight(&mut provider_state, &header2, &sig2, &meta, worst_case)
            .await
            .expect(
                "after one 402 round-trip + resync, the re-signed voucher must pass — \
                 this is the chat path that previously surfaced as auth_invalid",
            );

        // Provider's state advanced to exactly the new authorization.
        assert!(
            provider_state.cumulative_authorized_micros >= 200_001,
            "provider state should be at or above the previous floor after retry",
        );
    }
}
