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

use async_trait::async_trait;
use sdk::wallet_context::WalletContext;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ::types::base::SomaAddress;
use ::types::channel::{Channel, HttpVoucher, Voucher};
use ::types::crypto::GenericSignature;
use ::types::object::ObjectID;

use crate::channel::header::{decode_onchain_sig, encode_onchain_sig, SomaPayHeader};
use crate::channel::{ChannelError, PaymentChannel, RequestMeta};
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
}

mod onchain_sig_serde {
    use ::types::crypto::GenericSignature;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use fastcrypto::traits::ToFromBytes as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        v: &Option<GenericSignature>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
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
                let bytes = URL_SAFE_NO_PAD
                    .decode(s.as_bytes())
                    .map_err(serde::de::Error::custom)?;
                let sig = GenericSignature::from_bytes(&bytes)
                    .map_err(serde::de::Error::custom)?;
                Ok(Some(sig))
            }
        }
    }
}

impl TabProviderState {
    pub fn new(channel_id: ObjectID, chan: &Channel) -> Self {
        Self {
            channel_id,
            authorized_signer: chan.authorized_signer,
            deposit_micros: chan.deposit,
            cumulative_authorized_micros: chan.settled_amount,
            total_consumed_micros: chan.settled_amount,
            last_request_id: None,
            last_onchain_sig: None,
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
        Self {
            signing: None,
            clock_skew_tolerance_secs,
            auth_validity_secs: AUTH_VALIDITY_SECS,
        }
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

        // 3. Build the HttpVoucher and sign it (HTTP-bound layer).
        let body_sha256 = Self::body_sha256_from_hex(meta.body_sha256_hex)?;
        let request_id_sha = Self::request_id_sha(meta.request_id);
        let method_path_sha = Self::method_path_sha(meta.method, meta.path);
        let http_voucher = HttpVoucher::new(
            state.channel_id,
            new_cum,
            expires_ms,
            body_sha256,
            request_id_sha,
            method_path_sha,
        );
        let http_sig = sdk::channel::sign_http_voucher(
            &ctx.config.keystore,
            signer,
            &http_voucher,
        )
        .await
        .map_err(|e| ChannelError::Internal(format!("sign_http_voucher: {e}")))?;

        // 4. Sign the on-chain Voucher pair (cumulative+channel_id).
        //    The provider stores this for `Settle`.
        let onchain_sig = sdk::channel::sign_voucher(
            &ctx.config.keystore,
            signer,
            state.channel_id,
            new_cum,
        )
        .await
        .map_err(|e| ChannelError::Internal(format!("sign_voucher: {e}")))?;

        // 5. Commit local state.
        state.cumulative_authorized_micros = new_cum;
        if let Some(la) = state.last_authorized.as_ref() {
            state.realized.remove(&la.request_id);
        }
        state.last_authorized = Some(LastAuth {
            request_id: meta.request_id.to_string(),
            estimated_micros: worst_case_cost_micros,
        });

        let header = SomaPayHeader {
            channel_id: state.channel_id,
            http_voucher,
            http_sig,
        };
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
        meta: &RequestMeta<'_>,
        worst_case_cost_micros: u64,
    ) -> Result<(), ChannelError> {
        let header = SomaPayHeader::parse(header_value)?;

        // Expiry (with clock-skew tolerance).
        let now = now_ms();
        let tol_ms = self.clock_skew_tolerance_secs * 1000;
        if header.http_voucher.expires_ms + tol_ms < now {
            return Err(ChannelError::Expired);
        }
        if header.channel_id != state.channel_id {
            return Err(ChannelError::NotFound);
        }

        // HttpVoucher binding fields must match the actual request.
        let body_sha256 = Self::body_sha256_from_hex(meta.body_sha256_hex)?;
        if header.http_voucher.body_sha256 != body_sha256 {
            return Err(ChannelError::BadSignature);
        }
        if header.http_voucher.request_id_sha256 != Self::request_id_sha(meta.request_id) {
            return Err(ChannelError::BadSignature);
        }
        if header.http_voucher.method_path_sha256 != Self::method_path_sha(meta.method, meta.path) {
            return Err(ChannelError::BadSignature);
        }

        // Signature: build a synthetic Channel just for verification.
        // Only `authorized_signer` is read by `verify_http_voucher`.
        let channel = synthetic_channel(state.authorized_signer);
        sdk::channel::verify_http_voucher(&channel, &header.http_voucher, &header.http_sig)
            .map_err(|_| ChannelError::BadSignature)?;

        // Monotonic: same request id may re-present same cum;
        // otherwise must strictly increase.
        let same_request = state
            .last_request_id
            .as_deref()
            .map(|r| r == meta.request_id)
            .unwrap_or(false);
        let cum = header.http_voucher.cumulative_amount;
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

        state.cumulative_authorized_micros = cum;
        state.last_request_id = Some(meta.request_id.to_string());
        Ok(())
    }

    async fn post_flight(
        &self,
        state: &mut Self::ProviderState,
        _meta: &RequestMeta<'_>,
        actual_cost_micros: u64,
    ) -> Result<(), ChannelError> {
        state.total_consumed_micros = state
            .total_consumed_micros
            .saturating_add(actual_cost_micros);
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
    ) {
        state.realized.insert(request_id.to_string(), actual_cost_micros);
    }

    fn final_settlement(&self, state: &Self::ProviderState) -> Option<(Voucher, GenericSignature)> {
        state.last_onchain_sig.as_ref().map(|sig| {
            (
                Voucher::new(state.channel_id, state.cumulative_authorized_micros),
                sig.clone(),
            )
        })
    }
}

/// Build a stand-in `Channel` for signature verification — only
/// `authorized_signer` is read by `verify_http_voucher`. Avoids a
/// fresh chain read on every request.
fn synthetic_channel(authorized_signer: SomaAddress) -> Channel {
    Channel {
        payer: SomaAddress::ZERO,
        payee: SomaAddress::ZERO,
        authorized_signer,
        token: ::types::object::CoinType::Usdc,
        deposit: 0,
        settled_amount: 0,
        close_requested_at_ms: None,
    }
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
