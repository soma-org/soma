// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use serde_json::json;

use crate::pricing;
use crate::proxy::relay::{RelayCtx, forward_chat_completion};
use crate::proxy::router::Router as InnerRouter;

pub struct ClientState {
    pub router: Arc<InnerRouter>,
    pub relay: RelayCtx,
}

pub fn build_router(state: Arc<ClientState>, local_token: Option<String>) -> Router {
    let token_state = Arc::new(local_token);
    let protected = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(axum::middleware::from_fn_with_state(token_state, local_auth_middleware));
    Router::new().route("/health", get(health)).merge(protected).with_state(state)
}

/// Gate inbound requests on a shared bearer token. `None` disables the
/// check entirely (back-compat for callers that don't set
/// `Config.local_token`). `/health` is routed outside this layer so
/// supervisors can probe liveness without the secret.
pub async fn local_auth_middleware(
    State(token): State<Arc<Option<String>>>,
    req: Request,
    next: Next,
) -> Response {
    let Some(expected) = token.as_ref().as_deref() else {
        return next.run(req).await;
    };
    let provided = req
        .headers()
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(str::trim);
    if provided == Some(expected) {
        return next.run(req).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "error": {
                "message": "Invalid or missing local bearer token. Set Authorization: Bearer <SOMA_LOCAL_TOKEN>.",
                "type": "invalid_request_error",
                "code": "invalid_local_token",
            }
        })),
    )
        .into_response()
}

const MAX_BODY: usize = 16 * 1024 * 1024;

/// `GET /health` response. The first two fields (`status`,
/// `aggregate_healthy`) are kept for backward compatibility with
/// existing supervisors; the remainder is what the desktop watches to
/// surface serve-state and balance without re-querying the chain.
///
/// `balance_micros` is best-effort: it sums the headroom on every open
/// channel but does not include the raw wallet's free USDC balance
/// (which would require a chain RPC per probe). The desktop's
/// `soma balance --json` shell-out is still the source of truth for
/// total funds; this field exists so the proxy can flip
/// `can_serve` and emit a low-balance prompt without a chain probe.
#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    aggregate_healthy: bool,
    can_serve: bool,
    balance_micros: u64,
    wallet_address: String,
    channels_open: usize,
}

fn build_health_response(
    aggregate_healthy: bool,
    headrooms: &[u64],
    low_balance_threshold_micros: u64,
    wallet_address: Option<&str>,
) -> HealthResponse {
    let balance_micros: u64 = headrooms.iter().copied().fold(0u64, u64::saturating_add);
    // `can_serve` is the proactive cut-off: any channel with headroom
    // above the configured threshold means the next request will not
    // trigger the low-balance prompt. The hard 402 still fires when
    // `ensure_channel` literally cannot open — that's covered by I3.
    let can_serve = headrooms.iter().any(|h| *h > low_balance_threshold_micros);
    HealthResponse {
        status: "ok",
        aggregate_healthy,
        can_serve,
        balance_micros,
        wallet_address: wallet_address.unwrap_or("").to_string(),
        channels_open: headrooms.len(),
    }
}

async fn health(State(state): State<Arc<ClientState>>) -> impl IntoResponse {
    let healthy = state.router.aggregate_health().await;
    let headrooms = state.router.store.channel_headrooms().await;
    let body = build_health_response(
        healthy,
        &headrooms,
        state.router.cfg.low_balance_threshold_micros,
        state.router.cfg.wallet_address.as_deref(),
    );
    Json(body)
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

/// Build the structured 402 body for an insufficient-balance failure.
/// The message embeds the live balance, the wallet address (when the
/// proxy was configured with one), and a `soma://fund?...` deep-link
/// the desktop app's URL handler picks up. Shape matches OpenAI's
/// error envelope so off-the-shelf agents render it as a normal
/// error message instead of a transport failure.
pub(crate) fn insufficient_balance_body(
    available_micros: u64,
    wallet_address: Option<&str>,
) -> serde_json::Value {
    let mut msg = format!("Soma balance too low ({available_micros} micros available).");
    if let Some(addr) = wallet_address {
        msg.push_str(&format!(" Wallet: {addr}."));
    }
    msg.push_str(" Add USDC in the Soma Desktop app: soma://fund?reason=low_balance");
    json!({
        "error": {
            "type": "insufficient_balance",
            "code": "insufficient_balance",
            "message": msg,
        }
    })
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
        Err(e) => {
            return err(StatusCode::BAD_REQUEST, "bad_request", &format!("invalid chat body: {e}"));
        }
    };
    let model = chat.model.clone();
    let pick = match state.router.pick_provider_for_model(&model).await {
        Ok(Some(p)) => p,
        Ok(None) => {
            return err(
                StatusCode::NOT_FOUND,
                "no_provider",
                &format!("no provider exposes model {model}"),
            );
        }
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "router_error", &format!("{e}")),
    };
    let (provider, card) = pick;

    let slot = match state.router.ensure_channel(&provider, &model).await {
        Ok(s) => s,
        Err(e) => {
            if let Some(ib) =
                e.downcast_ref::<crate::proxy::router::InsufficientBalance>()
            {
                let body = insufficient_balance_body(
                    ib.available_micros,
                    state.router.cfg.wallet_address.as_deref(),
                );
                return (StatusCode::PAYMENT_REQUIRED, Json(body)).into_response();
            }
            // `{e:#}` walks the anyhow chain so the message includes both
            // the high-level context (e.g. "open_channel") and the
            // underlying chain failure ("ChannelTooManyOpenForPair { … }",
            // "ChannelOfferingMissing { … }", etc.). Default `{e}` only
            // shows the topmost context, which is useless for debugging.
            return err(StatusCode::INTERNAL_SERVER_ERROR, "channel_error", &format!("{e:#}"));
        }
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
        Err(e) => return err(StatusCode::BAD_GATEWAY, "upstream_error", &format!("{e}")),
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::ServiceExt;

    fn router_with_token(token: Option<String>) -> Router {
        let token_state = Arc::new(token);
        let protected = Router::new()
            .route("/v1/chat/completions", post(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(token_state, local_auth_middleware));
        Router::new().route("/health", get(|| async { "healthy" })).merge(protected)
    }

    async fn status(router: Router, builder: axum::http::request::Builder) -> StatusCode {
        let req = builder.body(Body::empty()).unwrap();
        router.oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn local_token_rejects_missing_bearer() {
        let r = router_with_token(Some("secret".into()));
        assert_eq!(
            status(r, Request::builder().uri("/v1/chat/completions").method("POST")).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn local_token_rejects_wrong_bearer() {
        let r = router_with_token(Some("secret".into()));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/v1/chat/completions")
                    .method("POST")
                    .header(http::header::AUTHORIZATION, "Bearer nope")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "invalid_local_token");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn local_token_accepts_correct_bearer() {
        let r = router_with_token(Some("secret".into()));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/v1/chat/completions")
                    .method("POST")
                    .header(http::header::AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn local_token_bypasses_health() {
        let r = router_with_token(Some("secret".into()));
        assert_eq!(
            status(r, Request::builder().uri("/health").method("GET")).await,
            StatusCode::OK
        );
    }

    /// 402 body carries the `insufficient_balance` envelope, the
    /// `soma://fund` deep-link, and the wallet address when configured
    /// — the message must render as a normal OpenAI error so
    /// off-the-shelf agents surface it to the user verbatim.
    #[test]
    fn insufficient_balance_body_includes_wallet_and_fund_link() {
        let v = insufficient_balance_body(123_456, Some("0xabc"));
        let err = &v["error"];
        assert_eq!(err["type"], "insufficient_balance");
        assert_eq!(err["code"], "insufficient_balance");
        let msg = err["message"].as_str().unwrap();
        assert!(msg.contains("123456 micros available"), "missing balance: {msg}");
        assert!(msg.contains("0xabc"), "missing wallet address: {msg}");
        assert!(msg.contains("soma://fund?reason=low_balance"), "missing deep link: {msg}");
    }

    /// Without a configured wallet address the message still renders;
    /// the wallet line is just omitted.
    #[test]
    fn insufficient_balance_body_omits_wallet_when_none() {
        let v = insufficient_balance_body(0, None);
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("0 micros available"));
        assert!(msg.contains("soma://fund?reason=low_balance"));
        assert!(!msg.contains("Wallet:"), "wallet line should be omitted: {msg}");
    }

    /// The handler's downcast path catches an `InsufficientBalance`
    /// error wrapped inside `anyhow::Error` — guards against a future
    /// refactor that adds extra `.context(...)` layers and silently
    /// breaks the typed match.
    #[test]
    fn insufficient_balance_downcast_round_trips_through_anyhow() {
        let ib = crate::proxy::router::InsufficientBalance {
            available_micros: 42,
            required_micros: 1_000,
        };
        let e: anyhow::Error = anyhow::Error::new(ib);
        let got = e
            .downcast_ref::<crate::proxy::router::InsufficientBalance>()
            .expect("downcast should succeed");
        assert_eq!(got.available_micros, 42);
        assert_eq!(got.required_micros, 1_000);
    }

    /// `/health` carries the old back-compat fields (`status`,
    /// `aggregate_healthy`) plus the desktop-facing serve-state
    /// (`can_serve`, `balance_micros`, `wallet_address`,
    /// `channels_open`). With headroom on at least one channel above
    /// the threshold, `can_serve` is true and the balance is the sum
    /// of all headrooms.
    #[test]
    fn health_response_includes_serve_state_and_balance() {
        let r =
            build_health_response(true, &[2_000_000, 500_000], 1_000_000, Some("0xabc"));
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["aggregate_healthy"], true);
        assert_eq!(v["can_serve"], true);
        assert_eq!(v["balance_micros"], 2_500_000u64);
        assert_eq!(v["wallet_address"], "0xabc");
        assert_eq!(v["channels_open"], 2);
    }

    /// When every channel's headroom is at or below the threshold
    /// (and the threshold is a *strict* lower bound), `can_serve`
    /// flips to false so the desktop raises the fund prompt before
    /// the next request hits the hard 402.
    #[test]
    fn health_response_flips_can_serve_below_threshold() {
        let r = build_health_response(false, &[100_000, 200_000], 1_000_000, None);
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["can_serve"], false);
        assert_eq!(v["balance_micros"], 300_000u64);
        assert_eq!(v["wallet_address"], "");
        assert_eq!(v["channels_open"], 2);
    }

    /// No channels open yet → zero balance, zero channels,
    /// `can_serve=false` (we don't know if a fresh open would
    /// succeed without a chain probe, so we surface false and let
    /// the desktop prompt for funding).
    #[test]
    fn health_response_no_channels() {
        let r = build_health_response(true, &[], 1_000_000, None);
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["can_serve"], false);
        assert_eq!(v["balance_micros"], 0u64);
        assert_eq!(v["channels_open"], 0);
    }

    #[tokio::test]
    async fn no_local_token_allows_all() {
        let r = router_with_token(None);
        assert_eq!(
            status(r.clone(), Request::builder().uri("/v1/chat/completions").method("POST")).await,
            StatusCode::OK
        );
        assert_eq!(
            status(r, Request::builder().uri("/health").method("GET")).await,
            StatusCode::OK
        );
    }
}
