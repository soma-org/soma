// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use crate::pricing;
use crate::proxy::relay::{forward_chat_completion, RelayCtx};
use crate::proxy::router::Router as InnerRouter;

pub struct ClientState {
    pub router: Arc<InnerRouter>,
    pub relay: RelayCtx,
}

pub fn build_router(state: Arc<ClientState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

const MAX_BODY: usize = 16 * 1024 * 1024;

async fn health(State(state): State<Arc<ClientState>>) -> impl IntoResponse {
    let healthy = state.router.aggregate_health().await;
    Json(json!({"status": "ok", "aggregate_healthy": healthy}))
}

async fn list_models(State(state): State<Arc<ClientState>>) -> impl IntoResponse {
    let map = state.router.aggregate_models().await;
    let mut data: Vec<serde_json::Value> = Vec::new();
    for (_id, (_p, c)) in map {
        if let Ok(v) = serde_json::to_value(&c) {
            data.push(v);
        }
    }
    Json(json!({"data": data}))
}

fn err(status: StatusCode, code: &str, msg: &str) -> Response {
    (status, Json(json!({"error":{"code":code,"message":msg}}))).into_response()
}

async fn chat_completions(
    State(state): State<Arc<ClientState>>,
    headers: HeaderMap,
    req: Request,
) -> Response {
    let bytes = match to_bytes(req.into_body(), MAX_BODY).await {
        Ok(b) => b,
        Err(_) => return err(StatusCode::BAD_REQUEST, "body_too_large", "body too large"),
    };
    let chat: crate::openai::ChatRequest = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => return err(
            StatusCode::BAD_REQUEST,
            "bad_request",
            &format!("invalid chat body: {e}"),
        ),
    };
    let model = chat.model.clone();
    let pick = match state.router.pick_provider_for_model(&model).await {
        Ok(Some(p)) => p,
        Ok(None) => return err(
            StatusCode::NOT_FOUND,
            "no_provider",
            &format!("no provider exposes model {model}"),
        ),
        Err(e) => return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "router_error",
            &format!("{e}"),
        ),
    };
    let (provider, card) = pick;

    let slot = match state.router.ensure_channel(&provider).await {
        Ok(s) => s,
        Err(e) => return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "channel_error",
            &format!("{e}"),
        ),
    };

    let worst_case = pricing::worst_case_for_request(&card, &chat);
    let is_stream = chat.stream.unwrap_or(false);

    let resp = match forward_chat_completion(
        &state.relay,
        &provider,
        &card,
        &slot,
        &headers,
        &bytes,
        worst_case,
        is_stream,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return err(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            &format!("{e}"),
        ),
    };

    if let Some(stream) = resp.stream {
        let mut builder = Response::builder().status(resp.status);
        for (k, v) in resp.headers.iter() {
            let n = k.as_str().to_ascii_lowercase();
            if matches!(n.as_str(), "content-type" | "cache-control" | "x-request-id") {
                builder = builder.header(k.clone(), v.clone());
            }
        }
        let body = Body::from_stream(stream);
        return builder.body(body).unwrap();
    }
    let bytes = resp.body_bytes.unwrap_or_default();
    let mut builder = Response::builder().status(resp.status);
    for (k, v) in resp.headers.iter() {
        let n = k.as_str().to_ascii_lowercase();
        if matches!(n.as_str(), "content-type" | "x-request-id") {
            builder = builder.header(k.clone(), v.clone());
        }
    }
    builder.body(Body::from(bytes)).unwrap()
}
