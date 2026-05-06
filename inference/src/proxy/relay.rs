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
use crate::channel::{PaymentChannel, RequestMeta, RunningTab};
use crate::http_util::pass_inbound;
use crate::new_request_id;
use crate::now_ms;
use crate::openai::stream::{extract_usage_from_chunk, find_double_newline};
use crate::pricing::realized_for_usage;
use crate::proxy::router::ProviderInfo;
use crate::proxy::state::ChannelSlot;

pub struct RelayCtx {
    pub channel: Arc<RunningTab>,
    pub http: reqwest::Client,
}

pub struct RelayedResponse {
    pub status: http::StatusCode,
    pub headers: HeaderMap,
    pub stream: Option<futures::stream::BoxStream<'static, Result<Bytes, std::io::Error>>>,
    pub body_bytes: Option<Bytes>,
}

async fn persist(slot: &Arc<tokio::sync::Mutex<ChannelSlot>>) {
    let g = slot.lock().await;
    let path = g.path.clone();
    if let Ok(s) = serde_json::to_string_pretty(&g.state) {
        let _ = std::fs::write(path, s);
    }
}

async fn reconcile_and_persist(
    channel: &RunningTab,
    slot: &Arc<tokio::sync::Mutex<ChannelSlot>>,
    request_id: &str,
    actual_micros: u64,
) {
    let mut g = slot.lock().await;
    channel.reconcile(&mut g.state, request_id, actual_micros).await;
    let path = g.path.clone();
    if let Ok(s) = serde_json::to_string_pretty(&g.state) {
        let _ = std::fs::write(path, s);
    }
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
        };

        let combined = {
            let mut g = slot.lock().await;
            ctx.channel
                .authorize(&mut g.state, &meta, worst_case_micros)
                .await
                .context("authorize")?
        };
        persist(slot).await;

        let url = format!("{}{}", provider.endpoint.trim_end_matches('/'), path);
        let mut h = pass_inbound(inbound_headers);
        // The combined value is `<somapay-header>||<onchain-sig-b64>`.
        // We send both in the Authorization header; the server splits.
        h.insert(http::header::AUTHORIZATION, combined.parse().unwrap());
        h.insert("x-request-id", request_id.parse().unwrap());
        if !h.contains_key(http::header::CONTENT_TYPE) {
            h.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        }

        let resp = ctx
            .http
            .post(&url)
            .headers(h)
            .body(inbound_body.clone())
            .send()
            .await?;

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
            persist(slot).await;
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
            tokio::spawn(async move {
                let mut buf: Vec<u8> = Vec::new();
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
                    if tx.send(Ok(bytes.clone())).await.is_err() {
                        return;
                    }
                    buf.extend_from_slice(&bytes);
                    while let Some(pos) = find_double_newline(&buf) {
                        let block: Vec<u8> = buf.drain(..pos + 2).collect();
                        let block_s = String::from_utf8_lossy(&block);
                        if let Some(u) = extract_usage_from_chunk(&block_s) {
                            let actual = realized_for_usage(&card, &u);
                            reconcile_and_persist(&channel, &slot, &rid, actual).await;
                        }
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
                if let Ok(usage) = serde_json::from_value::<crate::openai::Usage>(u.clone()) {
                    let actual = realized_for_usage(card, &usage);
                    reconcile_and_persist(&ctx.channel, slot, &request_id, actual).await;
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
