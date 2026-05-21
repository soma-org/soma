// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context as _;
use bytes::Bytes;
use futures::stream::StreamExt;
use http::HeaderMap;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::catalog::ModelCard;
use crate::channel::{PaymentChannel, RequestMeta, RequestUsage, RunningTab};
use crate::http_util::pass_inbound;
use crate::new_request_id;
use crate::now_ms;
use crate::openai::stream::{extract_usage_from_chunk, find_double_newline};
use crate::pricing::realized_for_usage;
use crate::proxy::router::ProviderInfo;
use crate::proxy::state::ChannelSlot;
use ::types::object::ObjectID;
use ::types::transaction::RatingReasonCode;

pub struct RelayCtx {
    pub channel: Arc<RunningTab>,
    pub http: reqwest::Client,
    /// Wallet used to submit `RateChannel(reason_code=...)` txs when the
    /// relay detects a TTFT or TTOT SLA breach. The payer is also the
    /// only allowed rater on-chain, so this is also the rating signer.
    pub wallet: Arc<sdk::wallet_context::WalletContext>,
    pub signer: ::types::base::SomaAddress,
}

pub struct RelayedResponse {
    pub status: http::StatusCode,
    pub headers: HeaderMap,
    pub stream: Option<futures::stream::BoxStream<'static, Result<Bytes, std::io::Error>>>,
    pub body_bytes: Option<Bytes>,
}

// Proxy state lives only in memory now — there is no per-request
// `persist` step. Callers that previously called `persist`/`reconcile_and_persist`
// just hold the lock long enough to update the in-memory `TabClientState`.
async fn reconcile(
    channel: &RunningTab,
    slot: &Arc<tokio::sync::Mutex<ChannelSlot>>,
    request_id: &str,
    actual_micros: u64,
    usage: RequestUsage,
) {
    let mut g = slot.lock().await;
    channel.reconcile(&mut g.state, request_id, actual_micros, usage).await;
}

/// Submit a `RateChannel(negative=true, reason_code=...)` for an SLA
/// breach, rate-limited to once per channel per epoch. Errors are
/// logged but never propagated back to the relayed response — a breach
/// rating is best-effort observability, not part of the request flow.
async fn emit_breach_rating(
    wallet: &Arc<sdk::wallet_context::WalletContext>,
    signer: ::types::base::SomaAddress,
    channel_id: ObjectID,
    reason: RatingReasonCode,
    slot: &Arc<tokio::sync::Mutex<ChannelSlot>>,
    last_emit_epoch: u64,
) {
    let current_epoch = match current_epoch_via_wallet(wallet).await {
        Some(e) => e,
        None => return,
    };
    if current_epoch <= last_emit_epoch && last_emit_epoch > 0 {
        // Already rated this channel for a breach in the current epoch —
        // skip silently; the next epoch's breach will fire again.
        return;
    }
    if let Err(e) = sdk::channel::rate(wallet, signer, channel_id, /*negative=*/ true, reason).await
    {
        tracing::warn!(channel = %channel_id, ?reason, err = %e, "RateChannel emission failed");
        return;
    }
    let mut g = slot.lock().await;
    g.state.last_breach_emitted_at_epoch = current_epoch;
}

async fn current_epoch_via_wallet(wallet: &Arc<sdk::wallet_context::WalletContext>) -> Option<u64> {
    let client = wallet.get_client().await.ok()?;
    let epoch_resp = client.get_epoch(None).await.ok()?;
    epoch_resp.epoch.and_then(|e| e.epoch)
}

pub async fn forward_chat_completion(
    ctx: &RelayCtx,
    provider: &ProviderInfo,
    card: &ModelCard,
    slot: &Arc<tokio::sync::Mutex<ChannelSlot>>,
    inbound_headers: &HeaderMap,
    inbound_body: &Bytes,
    worst_case_micros: u64,
    is_stream: bool,
) -> anyhow::Result<RelayedResponse> {
    let mut attempts = 0u32;
    let request_id = new_request_id();
    loop {
        attempts += 1;
        let body_sha = {
            let mut h = Sha256::new();
            h.update(inbound_body);
            hex::encode(h.finalize())
        };
        let path = "/v1/chat/completions";
        let meta = RequestMeta {
            method: "POST",
            path,
            body_sha256_hex: &body_sha,
            timestamp_ms: now_ms(),
            request_id: &request_id,
            model_id: &card.id,
        };

        let combined = {
            let mut g = slot.lock().await;
            ctx.channel
                .authorize(&mut g.state, &meta, worst_case_micros)
                .await
                .context("authorize")?
        };

        let url = format!("{}{}", provider.endpoint.trim_end_matches('/'), path);
        let mut h = pass_inbound(inbound_headers);
        // The combined value is `<somapay-header>||<onchain-sig-b64>`.
        // We send both in the Authorization header; the server splits.
        h.insert(http::header::AUTHORIZATION, combined.parse().unwrap());
        h.insert("x-request-id", request_id.parse().unwrap());
        if !h.contains_key(http::header::CONTENT_TYPE) {
            h.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        }

        // TTFT / TTOT measurement bookends. `t0` brackets the
        // upstream POST; `first_byte_ms` is sampled on the *first*
        // response chunk in the streaming path. For the non-streaming
        // path the "first byte" equals end-of-body, so TTFT/TTOT
        // collapse into a single latency.
        let t0 = now_ms();
        let resp = ctx.http.post(&url).headers(h).body(inbound_body.clone()).send().await?;

        let status = resp.status();
        let resp_headers = resp.headers().clone();

        if status.as_u16() == 402 && attempts == 1 {
            let need: u64 = resp_headers
                .get("x-soma-required-authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if need == 0 {
                anyhow::bail!("402 without x-soma-required-authorization");
            }
            let mut g = slot.lock().await;
            if need > g.state.deposit_micros {
                anyhow::bail!("402 needs {need} but deposit is {}", g.state.deposit_micros);
            }
            // Force the next signature to land at exactly `need`.
            g.state.cumulative_authorized_micros = need.saturating_sub(worst_case_micros);
            g.state.last_authorized = None;
            drop(g);
            continue;
        }

        if !status.is_success() {
            let body = resp.bytes().await.unwrap_or_default();
            return Ok(RelayedResponse {
                status,
                headers: resp_headers,
                stream: None,
                body_bytes: Some(body),
            });
        }

        if is_stream {
            let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
            let mut s = resp.bytes_stream();
            let card = card.clone();
            let slot = slot.clone();
            let channel = ctx.channel.clone();
            let rid = request_id.clone();
            let wallet = ctx.wallet.clone();
            let signer = ctx.signer;
            let channel_id = {
                let g = slot.lock().await;
                g.state.channel_id
            };
            tokio::spawn(async move {
                let mut buf: Vec<u8> = Vec::new();
                let mut first_byte_ms: Option<u64> = None;
                // Latest usage block seen on this stream — OpenAI-compatible
                // backends (incl. Anthropic via OpenRouter) often emit
                // *cumulative* usage in every `message_delta` event, not
                // a per-chunk delta, so summing them per event would
                // multi-count. Mirror the provider's `found_usage = Some(u)`
                // pattern: keep the most recent value, reconcile once
                // when the stream ends.
                let mut latest_usage: Option<crate::openai::Usage> = None;
                let mut last_byte_ms: u64 = now_ms();
                while let Some(item) = s.next().await {
                    let bytes = match item {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = tx
                                .send(Err(std::io::Error::new(std::io::ErrorKind::Other, e)))
                                .await;
                            return;
                        }
                    };
                    if first_byte_ms.is_none() {
                        first_byte_ms = Some(now_ms());
                    }
                    last_byte_ms = now_ms();
                    if tx.send(Ok(bytes.clone())).await.is_err() {
                        return;
                    }
                    buf.extend_from_slice(&bytes);
                    while let Some(pos) = find_double_newline(&buf) {
                        let block: Vec<u8> = buf.drain(..pos + 2).collect();
                        let block_s = String::from_utf8_lossy(&block);
                        if let Some(u) = extract_usage_from_chunk(&block_s) {
                            latest_usage = Some(u);
                        }
                    }
                }
                // Drain any tail bytes that didn't end on a `\n\n`
                // boundary — the upstream sometimes terminates the
                // final `data:` line without a trailing double-newline.
                if !buf.is_empty() {
                    let s = String::from_utf8_lossy(&buf);
                    if let Some(u) = extract_usage_from_chunk(&s) {
                        latest_usage = Some(u);
                    }
                }

                // Reconcile + SLA check exactly once per request, using
                // the last cumulative-usage report from the stream.
                if let Some(u) = latest_usage {
                    let actual = realized_for_usage(&card, &u);
                    let usage = RequestUsage::from_openai(&u);
                    reconcile(&channel, &slot, &rid, actual, usage).await;

                    // TTFT / TTOT against the channel's snapshotted SLA
                    // bounds. Same rate-limit + breach-rating shape as
                    // before, just lifted out of the per-chunk loop.
                    let ttft_ms = first_byte_ms.map(|fb| fb.saturating_sub(t0)).unwrap_or(0);
                    let completion_tokens = u.completion_tokens as u64;
                    let ttot_ms = if completion_tokens > 0 {
                        let stream_ms = last_byte_ms.saturating_sub(first_byte_ms.unwrap_or(t0));
                        stream_ms / completion_tokens
                    } else {
                        0
                    };
                    let g = slot.lock().await;
                    let ttft_bound = g.state.ttft_bound_ms;
                    let ttot_bound = g.state.ttot_bound_ms;
                    let last_emit_epoch = g.state.last_breach_emitted_at_epoch;
                    drop(g);
                    let ttft_breach = ttft_bound > 0 && ttft_ms > ttft_bound as u64;
                    let ttot_breach = ttot_bound > 0 && ttot_ms > ttot_bound as u64;
                    if ttft_breach || ttot_breach {
                        let reason = if ttft_breach {
                            RatingReasonCode::TtftBreach
                        } else {
                            RatingReasonCode::TtotBreach
                        };
                        tracing::warn!(
                            request_id = %rid,
                            ttft_ms,
                            ttot_ms,
                            ttft_bound,
                            ttot_bound,
                            ?reason,
                            "SLA breach"
                        );
                        emit_breach_rating(
                            &wallet,
                            signer,
                            channel_id,
                            reason,
                            &slot,
                            last_emit_epoch,
                        )
                        .await;
                    }
                }
            });
            return Ok(RelayedResponse {
                status,
                headers: resp_headers,
                stream: Some(ReceiverStream::new(rx).boxed()),
                body_bytes: None,
            });
        }

        // Non-streaming.
        let body = resp.bytes().await?;
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body) {
            if let Some(u) = v.get("usage") {
                if let Ok(openai_usage) = serde_json::from_value::<crate::openai::Usage>(u.clone())
                {
                    let actual = realized_for_usage(card, &openai_usage);
                    let usage = RequestUsage::from_openai(&openai_usage);
                    reconcile(&ctx.channel, slot, &request_id, actual, usage).await;
                }
            }
        }
        return Ok(RelayedResponse {
            status,
            headers: resp_headers,
            stream: None,
            body_bytes: Some(body),
        });
    }
}
