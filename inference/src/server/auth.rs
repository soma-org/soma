// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http::HeaderValue;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::chain::types::{ChannelHandle, ChannelStatus};
use crate::channel::header::SomaPayHeader;
use crate::channel::{ChannelError, PaymentChannel, RequestMeta};
use crate::new_request_id;
use crate::now_ms;
use crate::openai::ChatRequest;
use crate::pricing;
use crate::server::handler::ProviderState;

const MAX_BODY: usize = 16 * 1024 * 1024;

#[derive(Clone)]
pub struct PreparedRequest {
    pub request_id: String,
    pub body_sha256_hex: String,
    pub method: String,
    pub path: String,
    pub worst_case_micros: u64,
    pub model_id: String,
}

fn err_response(status: StatusCode, code: &str, msg: &str) -> Response {
    let body = json!({"error": {"code": code, "message": msg}});
    (status, axum::Json(body)).into_response()
}

pub async fn auth_middleware(
    state: axum::extract::State<Arc<ProviderState>>,
    req: Request,
    next: Next,
) -> Response {
    let (parts, body) = req.into_parts();
    let bytes = match to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(_) => {
            return err_response(StatusCode::BAD_REQUEST, "body_too_large", "body too large")
        }
    };

    let mut h = Sha256::new();
    h.update(&bytes);
    let body_sha256_hex = hex::encode(h.finalize());

    let chat: ChatRequest = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => return err_response(
            StatusCode::BAD_REQUEST,
            "bad_request",
            &format!("invalid chat body: {e}"),
        ),
    };

    let request_id = parts
        .headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(new_request_id);

    let card = match state.catalog.iter().find(|c| c.id == chat.model) {
        Some(c) => c.clone(),
        None => return err_response(
            StatusCode::BAD_REQUEST,
            "unknown_model",
            &format!("model not in catalog: {}", chat.model),
        ),
    };

    let worst_case = pricing::worst_case_for_request(&card, &chat);

    let path = parts.uri.path().to_string();
    let method = parts.method.as_str().to_string();
    let meta = RequestMeta {
        method: &method,
        path: &path,
        body_sha256_hex: &body_sha256_hex,
        timestamp_ms: now_ms(),
        request_id: &request_id,
    };

    let auth_header = match parts
        .headers
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        Some(s) => s,
        None => return err_response(
            StatusCode::UNAUTHORIZED,
            "auth_missing",
            "Authorization header missing",
        ),
    };

    let parsed = match SomaPayHeader::parse(auth_header) {
        Ok(p) => p,
        Err(_) => return err_response(
            StatusCode::UNAUTHORIZED,
            "auth_malformed",
            "SomaPay header malformed",
        ),
    };

    let handle = ChannelHandle(parsed.handle.clone());

    let chan_state = match state.chain.channel(&handle).await {
        Ok(s) => s,
        Err(_) => return err_response(
            StatusCode::NOT_FOUND,
            "channel_unknown",
            "channel handle unknown",
        ),
    };
    if chan_state.status != ChannelStatus::Open {
        return err_response(
            StatusCode::UNAUTHORIZED,
            "channel_closed",
            "channel not open",
        );
    }

    let slot = match state.ledger.slot(&handle).await {
        Some(s) => s,
        None => match state.ledger.init_slot(&chan_state).await {
            Ok(s) => s,
            Err(e) => return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "ledger_init_failed",
                &format!("{e}"),
            ),
        },
    };

    let guard = slot.lock_owned().await;
    let mut slot_inner = guard;
    let result = state
        .channel
        .pre_flight(&mut slot_inner.state, auth_header, &meta, worst_case)
        .await;
    match result {
        Ok(()) => {
            if let Err(e) = state.ledger.persist(&mut slot_inner).await {
                return err_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "ledger_persist_failed",
                    &format!("{e}"),
                );
            }
        }
        Err(ChannelError::Expired) => {
            return err_response(StatusCode::REQUEST_TIMEOUT, "expired", "header expired")
        }
        Err(ChannelError::PaymentRequired { need_micros }) => {
            let mut resp = err_response(
                StatusCode::PAYMENT_REQUIRED,
                "payment_required",
                "authorize a higher cumulative",
            );
            resp.headers_mut().insert(
                "x-soma-required-authorization",
                HeaderValue::from_str(&need_micros.to_string()).unwrap(),
            );
            return resp;
        }
        Err(ChannelError::NotFound) => {
            return err_response(
                StatusCode::NOT_FOUND,
                "channel_unknown",
                "channel handle unknown",
            )
        }
        Err(ChannelError::BadSignature)
        | Err(ChannelError::Malformed)
        | Err(ChannelError::NonMonotonic)
        | Err(ChannelError::OverDeposit) => {
            return err_response(StatusCode::UNAUTHORIZED, "auth_invalid", "auth invalid")
        }
        Err(other) => {
            return err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                &format!("{other}"),
            )
        }
    }

    let prep = PreparedRequest {
        request_id: request_id.clone(),
        body_sha256_hex,
        method,
        path,
        worst_case_micros: worst_case,
        model_id: chat.model.clone(),
    };

    let mut req = Request::from_parts(parts, Body::from(bytes));
    req.extensions_mut().insert(prep);
    req.extensions_mut().insert(chat);
    let guard_holder = SlotGuard(std::sync::Arc::new(std::sync::Mutex::new(Some(slot_inner))));
    req.extensions_mut().insert(guard_holder);
    let mut resp = next.run(req).await;
    resp.headers_mut().insert(
        "x-request-id",
        HeaderValue::from_str(&request_id).unwrap_or(HeaderValue::from_static("")),
    );
    resp
}

/// Holds the locked Slot guard so the handler can `take()` it, run
/// post_flight, persist, then drop.
#[derive(Clone)]
pub struct SlotGuard(
    pub std::sync::Arc<std::sync::Mutex<Option<tokio::sync::OwnedMutexGuard<crate::server::ledger::Slot>>>>,
);

impl SlotGuard {
    pub fn take(&self) -> Option<tokio::sync::OwnedMutexGuard<crate::server::ledger::Slot>> {
        self.0.lock().ok().and_then(|mut g| g.take())
    }
}
