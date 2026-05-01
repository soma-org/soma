// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cumulative-authorization "running tab" payment channel.
//!
//! Each request bumps the channel's cumulative authorized total and is
//! signed with the client's Ed25519 key. The provider verifies the latest
//! signature, runs the upstream call, charges realized cost against the
//! channel's consumed total, and could in principle present the latest
//! signed cumulative on-chain to claim that amount. Slack from over-
//! estimating the previous request's cost is absorbed into the next
//! request's authorization, so cost growth is approximately realized.
//!
//! Constraints enforced by [`RunningTab::pre_flight`]:
//! - signature valid under the client's pubkey
//! - cumulative monotonic across requests
//! - cumulative ≤ deposit (over-deposit is rejected)
//! - cumulative ≥ already-consumed + worst-case (otherwise [`ChannelError::PaymentRequired`])

use std::collections::HashMap;

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use fastcrypto::ed25519::{Ed25519KeyPair, Ed25519PublicKey, Ed25519Signature};
use fastcrypto::traits::{KeyPair, Signer, ToFromBytes, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use types::base::SomaAddress;

use crate::channel::header::{digest_input, SomaPayHeader};
use crate::channel::{ChannelError, PaymentChannel, RequestMeta};
use crate::chain::types::ChannelHandle;
use crate::now_ms;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LastAuth {
    pub request_id: String,
    pub estimated_micros: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TabClientState {
    pub handle: ChannelHandle,
    pub provider_address: SomaAddress,
    pub provider_pubkey_hex: String,
    pub provider_endpoint: String,
    pub deposit_micros: u64,
    pub cumulative_authorized_micros: u64,
    pub last_authorized: Option<LastAuth>,
    /// Map of request_id -> realized cost. Entry is consumed by the next
    /// `authorize` call that uses it.
    #[serde(default)]
    pub realized: HashMap<String, u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TabProviderState {
    pub handle: ChannelHandle,
    pub client_address: SomaAddress,
    pub client_pubkey_hex: String,
    pub deposit_micros: u64,
    pub cumulative_authorized_micros: u64,
    pub total_consumed_micros: u64,
    pub last_signature_b64: String,
    pub last_request_id: Option<String>,
}

pub struct RunningTab {
    /// Set on the client side; absent on the provider side.
    pub signing_key: Option<Ed25519KeyPair>,
    pub clock_skew_tolerance_secs: u64,
    pub auth_validity_secs: u64,
}

impl RunningTab {
    pub fn for_provider(clock_skew_tolerance_secs: u64) -> Self {
        Self {
            signing_key: None,
            clock_skew_tolerance_secs,
            auth_validity_secs: 60,
        }
    }

    pub fn for_client(signing: Ed25519KeyPair) -> Self {
        Self {
            signing_key: Some(signing),
            clock_skew_tolerance_secs: 60,
            auth_validity_secs: 60,
        }
    }
}

fn verifying_key_from_hex(hex_pub: &str) -> Result<Ed25519PublicKey, ChannelError> {
    let bytes = hex::decode(hex_pub).map_err(|_| ChannelError::Internal("bad pubkey hex".into()))?;
    Ed25519PublicKey::from_bytes(&bytes).map_err(|_| ChannelError::Internal("bad pubkey".into()))
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
        let signing = self
            .signing_key
            .as_ref()
            .ok_or_else(|| ChannelError::Internal("no signing key".into()))?;

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

        // 2. Bump. Ensure strictly positive so cumulative advances even when
        //    accumulated slack would otherwise zero out the bump.
        let bump_raw = worst_case_cost_micros.saturating_sub(slack);
        let bump = bump_raw.max(1);
        let new_cum = state.cumulative_authorized_micros.saturating_add(bump);
        if new_cum > state.deposit_micros {
            return Err(ChannelError::Internal(
                "would exceed deposit; soma cli top-up needed".into(),
            ));
        }
        let expires_ms = now_ms() + self.auth_validity_secs * 1000;

        let bytes = digest_input(
            &state.handle.0,
            new_cum,
            expires_ms,
            meta.method,
            meta.path,
            meta.body_sha256_hex,
            meta.request_id,
        );
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let digest = hasher.finalize();
        let sig: Ed25519Signature = signing.sign(&digest);
        let sig_b64 = URL_SAFE_NO_PAD.encode(sig.as_bytes());

        // Persist (caller flushes to disk).
        state.cumulative_authorized_micros = new_cum;
        if let Some(la) = state.last_authorized.as_ref() {
            state.realized.remove(&la.request_id);
        }
        state.last_authorized = Some(LastAuth {
            request_id: meta.request_id.to_string(),
            estimated_micros: worst_case_cost_micros,
        });

        let header = SomaPayHeader {
            handle: state.handle.0.clone(),
            cum_micros: new_cum,
            expires_ms,
            sig_b64,
        };
        Ok(header.format())
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
        if header.expires_ms + tol_ms < now {
            return Err(ChannelError::Expired);
        }
        // Caller is responsible for selecting the matching state.
        if header.handle != state.handle.0 {
            return Err(ChannelError::NotFound);
        }
        // Signature.
        let bytes = digest_input(
            &header.handle,
            header.cum_micros,
            header.expires_ms,
            meta.method,
            meta.path,
            meta.body_sha256_hex,
            meta.request_id,
        );
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let digest = hasher.finalize();
        let vk = verifying_key_from_hex(&state.client_pubkey_hex)?;
        let sig_bytes = header.sig_bytes()?;
        let sig = Ed25519Signature::from_bytes(&sig_bytes)
            .map_err(|_| ChannelError::Malformed)?;
        vk.verify(&digest, &sig).map_err(|_| ChannelError::BadSignature)?;
        // Monotonic: same request id may re-present same cum; otherwise must strictly increase.
        let same_request = state
            .last_request_id
            .as_deref()
            .map(|r| r == meta.request_id)
            .unwrap_or(false);
        if header.cum_micros < state.cumulative_authorized_micros {
            return Err(ChannelError::NonMonotonic);
        }
        if header.cum_micros == state.cumulative_authorized_micros && !same_request {
            return Err(ChannelError::NonMonotonic);
        }
        // Deposit cap.
        if header.cum_micros > state.deposit_micros {
            return Err(ChannelError::OverDeposit);
        }
        // Payment required: the new authorization must cover already-consumed + worst-case.
        let need = state.total_consumed_micros.saturating_add(worst_case_cost_micros);
        if header.cum_micros < need {
            return Err(ChannelError::PaymentRequired { need_micros: need });
        }
        state.cumulative_authorized_micros = header.cum_micros;
        state.last_signature_b64 = header.sig_b64.clone();
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
                handle = %state.handle,
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

    fn final_settlement(&self, state: &Self::ProviderState) -> (u64, String) {
        (state.cumulative_authorized_micros, state.last_signature_b64.clone())
    }
}

#[cfg(test)]
mod tests {
    use fastcrypto::ed25519::Ed25519KeyPair;
    use fastcrypto::traits::KeyPair;

    use super::*;

    fn meta<'a>(rid: &'a str, body: &'a str) -> RequestMeta<'a> {
        RequestMeta {
            method: "POST",
            path: "/v1/chat/completions",
            body_sha256_hex: body,
            timestamp_ms: now_ms(),
            request_id: rid,
        }
    }

    #[tokio::test]
    async fn three_request_session() {
        let kp = Ed25519KeyPair::generate(&mut rand::thread_rng());
        let pubkey_hex = hex::encode(kp.public().as_bytes());
        let signing = kp.copy();
        let handle = ChannelHandle("01HZZ".to_string());
        let deposit = 1_000_000_u64; // $1
        let mut client_state = TabClientState {
            handle: handle.clone(),
            provider_address: SomaAddress::ZERO,
            provider_pubkey_hex: "00".repeat(32),
            provider_endpoint: "http://x".into(),
            deposit_micros: deposit,
            cumulative_authorized_micros: 0,
            last_authorized: None,
            realized: HashMap::new(),
        };
        let mut provider_state = TabProviderState {
            handle: handle.clone(),
            client_address: SomaAddress::ZERO,
            client_pubkey_hex: pubkey_hex.clone(),
            deposit_micros: deposit,
            cumulative_authorized_micros: 0,
            total_consumed_micros: 0,
            last_signature_b64: String::new(),
            last_request_id: None,
        };

        let client = RunningTab::for_client(signing);
        let provider = RunningTab::for_provider(60);

        // R1: estimate $0.10 = 100_000 micros, post-flight $0.07 = 70_000.
        let r1 = "req1";
        let m1 = meta(r1, "aa");
        let h1 = client.authorize(&mut client_state, &m1, 100_000).await.unwrap();
        provider.pre_flight(&mut provider_state, &h1, &m1, 100_000).await.unwrap();
        provider.post_flight(&mut provider_state, &m1, 70_000).await.unwrap();
        client.reconcile(&mut client_state, r1, 70_000).await;
        assert_eq!(client_state.cumulative_authorized_micros, 100_000);

        // R2: estimate $0.05 = 50_000; slack = 30_000; bump = 20_000.
        let r2 = "req2";
        let m2 = meta(r2, "bb");
        let h2 = client.authorize(&mut client_state, &m2, 50_000).await.unwrap();
        provider.pre_flight(&mut provider_state, &h2, &m2, 50_000).await.unwrap();
        provider.post_flight(&mut provider_state, &m2, 40_000).await.unwrap();
        client.reconcile(&mut client_state, r2, 40_000).await;
        assert_eq!(client_state.cumulative_authorized_micros, 120_000);

        // R3: way over deposit.
        let r3 = "req3";
        let m3 = meta(r3, "cc");
        let err = client.authorize(&mut client_state, &m3, 99_000_000).await.unwrap_err();
        match err {
            ChannelError::Internal(s) => assert!(s.contains("exceed deposit")),
            other => panic!("expected internal, got {other:?}"),
        }

        // PaymentRequired path: total_consumed=110_000, current cum=120_000.
        // For worst-case 50_000, need = 160_000. Sign at cum=150_000 (under need).
        let r4 = "req4";
        let m4 = meta(r4, "dd");
        let mut tmp_client = client_state.clone();
        tmp_client.cumulative_authorized_micros = 100_000; // back off so authorize bumps to 150_000
        let h4 = client.authorize(&mut tmp_client, &m4, 50_000).await.unwrap();
        let res = provider.pre_flight(&mut provider_state, &h4, &m4, 50_000).await.unwrap_err();
        match res {
            ChannelError::PaymentRequired { need_micros } => {
                assert_eq!(need_micros, 110_000 + 50_000);
            }
            other => panic!("expected PaymentRequired, got {other:?}"),
        }

        let (cum, _sig) = provider.final_settlement(&provider_state);
        assert_eq!(cum, 120_000);
    }
}
