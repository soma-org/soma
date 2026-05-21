// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::catalog::ModelCard;
use crate::chain::ChannelSurface;
use crate::channel::{PaymentChannel, RunningTab};
use crate::now_ms;
use crate::openai::stream::{extract_usage_from_chunk, find_double_newline};
use crate::pricing;
use crate::server::auth::{PreparedRequest, SlotGuard, auth_middleware};
use crate::server::backend::Backend;
use crate::server::ledger::Ledger;

pub struct ProviderState {
    pub chain: Arc<dyn ChannelSurface>,
    pub backend: Arc<dyn Backend>,
    pub channel: Arc<RunningTab>,
    pub ledger: Ledger,
    /// In-memory copy of the operator's TOML `[[offerings]]`. The auth
    /// middleware looks up the matching card by model id to validate
    /// the request against its pricing + SLA; `chat_completions` uses
    /// the same card for realized-cost calculation. The proxy no
    /// longer reads this over HTTP — it gets the same data from the
    /// indexer's `offerings()` query.
    pub catalog: Vec<ModelCard>,
}

pub fn build_router(state: Arc<ProviderState>) -> Router {
    // Only two HTTP surfaces remain: the SomaPay-authorized chat
    // endpoint, and a liveness probe. Discovery (provider record,
    // model catalog, channel state) is indexer-only — `/soma/info`,
    // `/v1/models`, and `/soma/channel/{id}` are gone.
    let auth_state = state.clone();
    let v1 = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .layer(axum::middleware::from_fn_with_state(auth_state, auth_middleware));

    Router::new().route("/health", get(health)).merge(v1).with_state(state)
}

async fn health(State(state): State<Arc<ProviderState>>) -> impl IntoResponse {
    let healthy = state.backend.health().await;
    Json(json!({"status": "ok", "backend_healthy": healthy}))
}

async fn chat_completions(
    State(state): State<Arc<ProviderState>>,
    headers: HeaderMap,
    Extension(prep): Extension<PreparedRequest>,
    Extension(chat): Extension<crate::openai::ChatRequest>,
    maybe_slot: Option<Extension<SlotGuard>>,
) -> Response {
    let stream = chat.stream.unwrap_or(false);
    let slot = maybe_slot.map(|e| e.0);
    if stream {
        run_streaming(state, headers, prep, chat, slot).await
    } else {
        run_non_streaming(state, headers, prep, chat, slot).await
    }
}

async fn run_non_streaming(
    state: Arc<ProviderState>,
    headers: HeaderMap,
    prep: PreparedRequest,
    chat: crate::openai::ChatRequest,
    slot: Option<SlotGuard>,
) -> Response {
    let card = state.catalog.iter().find(|c| c.id == prep.model_id).cloned();
    let card = match card {
        Some(c) => c,
        None => return error(StatusCode::BAD_REQUEST, "unknown_model", "model not in catalog"),
    };
    match state.backend.chat_completions(chat, headers).await {
        Ok(v) => {
            let usage_val = v.get("usage").cloned();
            let parsed_usage = usage_val
                .as_ref()
                .and_then(|u| serde_json::from_value::<crate::openai::Usage>(u.clone()).ok());
            let actual =
                parsed_usage.as_ref().map(|u| pricing::realized_for_usage(&card, u)).unwrap_or(0);
            let usage_breakdown = parsed_usage
                .as_ref()
                .map(crate::channel::RequestUsage::from_openai)
                .unwrap_or_default();
            if let Some(holder) = slot {
                if let Some(mut guard) = holder.take() {
                    let meta = crate::channel::RequestMeta {
                        method: &prep.method,
                        path: &prep.path,
                        body_sha256_hex: &prep.body_sha256_hex,
                        timestamp_ms: now_ms(),
                        request_id: &prep.request_id,
                        model_id: &prep.model_id,
                    };
                    let _ = state
                        .channel
                        .post_flight(&mut guard.state, &meta, actual, usage_breakdown)
                        .await;
                    let _ = state.ledger.persist(&mut *guard).await;
                }
            }
            let mut resp = (StatusCode::OK, Json(v)).into_response();
            resp.headers_mut().insert(
                HeaderName::from_static("x-request-id"),
                HeaderValue::from_str(&prep.request_id).unwrap_or(HeaderValue::from_static("")),
            );
            resp
        }
        Err(e) => error(StatusCode::BAD_GATEWAY, "upstream_error", &format!("{e}")),
    }
}

async fn run_streaming(
    state: Arc<ProviderState>,
    headers: HeaderMap,
    prep: PreparedRequest,
    chat: crate::openai::ChatRequest,
    slot: Option<SlotGuard>,
) -> Response {
    let card = state.catalog.iter().find(|c| c.id == prep.model_id).cloned();
    let card = match card {
        Some(c) => c,
        None => return error(StatusCode::BAD_REQUEST, "unknown_model", "model not in catalog"),
    };

    let upstream = match state.backend.chat_completions_stream(chat, headers).await {
        Ok(s) => s,
        Err(e) => return error(StatusCode::BAD_GATEWAY, "upstream_error", &format!("{e}")),
    };

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let state_for_task = state.clone();
    let prep_for_task = prep.clone();
    let card_for_task = card.clone();

    let mut owned_slot = slot.and_then(|s| s.take());

    tokio::spawn(async move {
        let mut s = upstream;
        let mut buf: Vec<u8> = Vec::new();
        let mut found_usage: Option<crate::openai::Usage> = None;
        while let Some(item) = s.next().await {
            let bytes = match item {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::new(std::io::ErrorKind::Other, e))).await;
                    break;
                }
            };
            if tx.send(Ok(bytes.clone())).await.is_err() {
                break;
            }
            buf.extend_from_slice(&bytes);
            while let Some(pos) = find_double_newline(&buf) {
                let block = buf.drain(..pos + 2).collect::<Vec<u8>>();
                let s = String::from_utf8_lossy(&block);
                if let Some(u) = extract_usage_from_chunk(&s) {
                    found_usage = Some(u);
                }
            }
        }
        if !buf.is_empty() {
            let s = String::from_utf8_lossy(&buf);
            if let Some(u) = extract_usage_from_chunk(&s) {
                found_usage = Some(u);
            }
        }
        if let Some(mut guard) = owned_slot.take() {
            let actual = match &found_usage {
                Some(u) => pricing::realized_for_usage(&card_for_task, u),
                None => {
                    tracing::warn!(
                        request_id = %prep_for_task.request_id,
                        "no usage chunk found; charging worst-case"
                    );
                    prep_for_task.worst_case_micros
                }
            };
            let usage_breakdown = found_usage
                .as_ref()
                .map(crate::channel::RequestUsage::from_openai)
                .unwrap_or_default();
            let meta = crate::channel::RequestMeta {
                method: &prep_for_task.method,
                path: &prep_for_task.path,
                body_sha256_hex: &prep_for_task.body_sha256_hex,
                timestamp_ms: now_ms(),
                request_id: &prep_for_task.request_id,
                model_id: &prep_for_task.model_id,
            };
            let _ = state_for_task
                .channel
                .post_flight(&mut guard.state, &meta, actual, usage_breakdown)
                .await;
            let _ = state_for_task.ledger.persist(&mut *guard).await;
            tracing::info!(
                request_id = %prep_for_task.request_id,
                actual_micros = actual,
                consumed_micros = guard.state.total_consumed_micros,
                "post_flight"
            );
        }
    });

    let body = Body::from_stream(ReceiverStream::new(rx));
    let mut resp = Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .header("x-request-id", &prep.request_id)
        .body(body)
        .unwrap();
    resp.headers_mut().insert(
        HeaderName::from_static("x-request-id"),
        HeaderValue::from_str(&prep.request_id).unwrap_or(HeaderValue::from_static("")),
    );
    resp
}

fn error(status: StatusCode, code: &str, msg: &str) -> Response {
    (status, Json(json!({"error":{"code":code,"message":msg}}))).into_response()
}
