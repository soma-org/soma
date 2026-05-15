// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared HTTP utilities — header passthrough and stream-options injection.

use http::HeaderMap;
use serde_json::Value;

use crate::openai::ChatRequest;

const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "authorization",
    "accept-encoding",
];

/// Filter inbound client headers down to the subset that's safe to forward
/// upstream. Drops hop-by-hop, host, content-length, authorization, and any
/// `x-soma-*` administrative headers.
pub fn pass_outbound(inbound: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (k, v) in inbound.iter() {
        let name = k.as_str().to_ascii_lowercase();
        if HOP_BY_HOP.contains(&name.as_str()) || name.starts_with("x-soma") {
            continue;
        }
        out.insert(k.clone(), v.clone());
    }
    out
}

/// Same as [`pass_outbound`] but also drops `x-request-id` (the proxy injects
/// its own).
pub fn pass_inbound(inbound: &HeaderMap) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (k, v) in inbound.iter() {
        let n = k.as_str().to_ascii_lowercase();
        if HOP_BY_HOP.contains(&n.as_str()) || n == "x-request-id" {
            continue;
        }
        h.insert(k.clone(), v.clone());
    }
    h
}

/// Force `stream_options.include_usage = true` so the upstream emits the
/// final usage chunk we need for post-flight reconciliation.
pub fn ensure_stream_options_include_usage(req: &mut ChatRequest) {
    if !matches!(req.stream, Some(true)) {
        return;
    }
    let so = req
        .stream_options
        .clone()
        .unwrap_or_else(|| serde_json::json!({}));
    let mut so_obj = match so {
        Value::Object(m) => m,
        _ => serde_json::Map::new(),
    };
    so_obj.insert("include_usage".to_string(), Value::Bool(true));
    req.stream_options = Some(Value::Object(so_obj));
}
