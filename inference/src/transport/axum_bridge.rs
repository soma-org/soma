// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bridge from the iroh transport onto an `axum::Router`.
//!
//! The provider's entire HTTP stack — `auth_middleware` (voucher verify),
//! the backend call, `post_flight`, the ledger — lives behind an
//! [`axum::Router`] (`server::handler::build_router`). Rather than
//! reimplement any of it for iroh, this bridge turns each inbound iroh
//! request into an `http::Request`, runs it through that same router via
//! [`tower::ServiceExt::oneshot`], and streams the `http::Response` back
//! over the iroh stream. So the provider speaks exactly the same OpenAI +
//! SomaPay semantics whether reached over HTTP or iroh.
//!
//! The authenticated peer [`EndpointId`] is attached to the request as the
//! [`PeerEndpointId`] extension, so the auth layer can enforce
//! `peer == channel.authorized_signer` (see the binding decision) without
//! the transport knowing anything about channels.

use std::io;

use axum::body::Body;
use bytes::Bytes;
use futures::stream::StreamExt;
use http::{HeaderName, HeaderValue};
use tower::ServiceExt;

use super::{EndpointId, IrohHandler, WireRequestMeta, WireResponse};

/// Request extension carrying the iroh-authenticated peer identity. Present
/// on every request that arrived over the iroh transport; absent for plain
/// HTTP. The provider's auth middleware reads it to bind the transport key
/// to the channel's `authorized_signer`.
#[derive(Debug, Clone, Copy)]
pub struct PeerEndpointId(pub EndpointId);

/// An [`IrohHandler`] that dispatches each request to an `axum::Router`.
#[derive(Clone)]
pub struct AxumBridge {
    router: axum::Router,
}

impl AxumBridge {
    pub fn new(router: axum::Router) -> Self {
        AxumBridge { router }
    }
}

fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

#[async_trait::async_trait]
impl IrohHandler for AxumBridge {
    async fn handle(&self, peer: EndpointId, req: WireRequestMeta, body: Bytes) -> WireResponse {
        // Reconstruct an http::Request from the wire framing.
        let mut request = http::Request::new(Body::from(body));
        *request.method_mut() =
            http::Method::from_bytes(req.method.as_bytes()).unwrap_or(http::Method::POST);
        *request.uri_mut() = req.path.parse().unwrap_or_else(|_| "/".parse().unwrap());
        for (k, v) in &req.headers {
            if let (Ok(name), Ok(val)) =
                (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(v))
            {
                request.headers_mut().insert(name, val);
            }
        }
        // Hand the authenticated peer key to the auth layer.
        request.extensions_mut().insert(PeerEndpointId(peer));

        // Run it through the real provider router. `Router`'s Service error
        // is `Infallible`, so this never errors.
        let response = match self.router.clone().oneshot(request).await {
            Ok(r) => r,
            Err(e) => {
                return WireResponse::bytes(
                    502,
                    "text/plain",
                    Bytes::from(format!("bridge error: {e}")),
                );
            }
        };

        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str().to_string(), v.to_string())))
            .collect();

        // Stream the response body chunk-by-chunk so SSE flushes incrementally.
        let body = response.into_body().into_data_stream().map(|r| r.map_err(io_err)).boxed();
        WireResponse::stream(status, headers, body)
    }
}
