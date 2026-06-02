// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! iroh P2P transport for the buyer → provider hop.
//!
//! Today the proxy reaches a provider over HTTP: `relay.rs` does a
//! `reqwest` `POST` to the provider's registered `endpoint` URL. This
//! module is the drop-in replacement for *only that hop*: the buyer dials
//! the provider by its **iroh [`EndpointId`]** (an Ed25519 public key) and
//! tunnels the same OpenAI-style request/response — both non-streaming and
//! streaming (SSE) — over a single bidirectional QUIC stream.
//!
//! Nothing about the local agent → proxy side changes: that stays plain
//! loopback HTTP. And the voucher accounting / SSE parsing / SLA logic in
//! the relay is transport-agnostic — it operates on the response bytes,
//! which arrive here as a [`BuyerResponse`] body stream just like they did
//! from `reqwest`.
//!
//! - [`IrohBuyer`] — the proxy/client side. [`IrohBuyer::post`] opens a
//!   bi-stream, writes the request, and returns a streamed response.
//! - [`IrohProvider`] — the provider/server side. It registers a single
//!   ALPN ([`SOMA_INFERENCE_ALPN`]) and dispatches each connection to an
//!   [`IrohHandler`]. The handler is handed the authenticated peer
//!   [`EndpointId`], so an allowlist (e.g. "must be the payer of an open
//!   channel") can be enforced at the transport boundary.

pub mod axum_bridge;
pub mod identity;
pub mod wire;

use std::io;
use std::sync::Arc;

use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use iroh::Endpoint;
use iroh::endpoint::{Connection, presets};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};

pub use axum_bridge::{AxumBridge, PeerEndpointId};
pub use identity::{iroh_secret_from_keypair, soma_address_from_endpoint_id};
pub use iroh::{EndpointAddr, EndpointId};
pub use wire::{WireRequestMeta, WireResponseMeta};

/// ALPN for the Soma inference request/response protocol. Bumped if the
/// wire framing in [`wire`] ever changes incompatibly.
pub const SOMA_INFERENCE_ALPN: &[u8] = b"soma/inference/1";

fn io_err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e.to_string())
}

/// Bind a loopback-only iroh endpoint (no relay, no discovery) — used by the
/// `bind_local*` constructors and tests. `with_alpn` declares the inference
/// ALPN for accepting endpoints; buyers pass `false`.
async fn build_local_endpoint(with_alpn: bool) -> anyhow::Result<Endpoint> {
    let mut builder = Endpoint::builder(presets::N0DisableRelay);
    if with_alpn {
        builder = builder.alpns(vec![SOMA_INFERENCE_ALPN.to_vec()]);
    }
    Ok(builder.bind_addr("127.0.0.1:0")?.bind().await?)
}

/// Build a production iroh endpoint (n0 relays + discovery) from a fixed
/// secret key, so its [`EndpointId`] is stable across restarts. `with_alpn`
/// declares the inference ALPN for accepting (provider) endpoints.
pub async fn build_n0_endpoint(
    secret: iroh::SecretKey,
    with_alpn: bool,
) -> anyhow::Result<Endpoint> {
    let mut builder = Endpoint::builder(presets::N0).secret_key(secret);
    if with_alpn {
        builder = builder.alpns(vec![SOMA_INFERENCE_ALPN.to_vec()]);
    }
    Ok(builder.bind().await?)
}

/// Load a persisted iroh secret key (hex) from `path`, or generate a fresh
/// one and persist it. A stable secret keeps the provider's [`EndpointId`]
/// constant across restarts, so its on-chain registration stays valid.
pub fn load_or_create_secret(path: &std::path::Path) -> anyhow::Result<iroh::SecretKey> {
    if let Ok(contents) = std::fs::read_to_string(path) {
        let bytes = hex::decode(contents.trim())
            .map_err(|e| anyhow::anyhow!("iroh key file {}: {e}", path.display()))?;
        let arr: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("iroh key file {} must be 32 bytes", path.display()))?;
        return Ok(iroh::SecretKey::from_bytes(&arr));
    }
    let secret = iroh::SecretKey::generate();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, hex::encode(secret.to_bytes()))?;
    Ok(secret)
}

/// The canonical (z-base-32) string form of an [`EndpointId`], as stored on
/// chain in `Provider.iroh_endpoint_id` and dialed by buyers.
pub fn endpoint_id_string(id: &EndpointId) -> String {
    id.to_string()
}

/// Build an [`EndpointAddr`] from a canonical `EndpointId` string plus
/// optional direct socket addresses. With no direct addresses the buyer
/// resolves the key via discovery (production); with direct addresses it
/// dials them straight (same-host / tests).
pub fn dial_addr(
    endpoint_id: &str,
    direct_addrs: &[std::net::SocketAddr],
) -> anyhow::Result<EndpointAddr> {
    let id: EndpointId = endpoint_id
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid endpoint id {endpoint_id:?}: {e}"))?;
    let mut addr = EndpointAddr::new(id);
    for sock in direct_addrs {
        addr = addr.with_ip_addr(*sock);
    }
    Ok(addr)
}

// ---------------------------------------------------------------------------
// Provider (server) side
// ---------------------------------------------------------------------------

/// A response produced by an [`IrohHandler`]. The body is a byte stream so a
/// streaming (SSE) completion and a one-shot JSON response share one type —
/// the provider just forwards `body` to the wire until it ends.
pub struct WireResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: BoxStream<'static, io::Result<Bytes>>,
}

impl WireResponse {
    /// A single-chunk response with an explicit content type.
    pub fn bytes(status: u16, content_type: &str, body: Bytes) -> Self {
        WireResponse {
            status,
            headers: vec![("content-type".to_string(), content_type.to_string())],
            body: stream::once(async move { Ok(body) }).boxed(),
        }
    }

    /// A one-shot `application/json` response.
    pub fn json(status: u16, value: &serde_json::Value) -> Self {
        let bytes = Bytes::from(serde_json::to_vec(value).unwrap_or_default());
        Self::bytes(status, "application/json", bytes)
    }

    /// A streaming response (e.g. `text/event-stream`) whose body is yielded
    /// chunk by chunk.
    pub fn stream(
        status: u16,
        headers: Vec<(String, String)>,
        body: BoxStream<'static, io::Result<Bytes>>,
    ) -> Self {
        WireResponse { status, headers, body }
    }
}

/// Application logic behind an [`IrohProvider`]. One call per inbound
/// request. `peer` is the authenticated [`EndpointId`] of the buyer — use it
/// to authorize (the on-chain `Channel` names the `payer`/`authorized_signer`
/// whose key this is).
#[async_trait::async_trait]
pub trait IrohHandler: Send + Sync + 'static {
    async fn handle(&self, peer: EndpointId, req: WireRequestMeta, body: Bytes) -> WireResponse;
}

/// A running provider: an iroh [`Router`] accepting [`SOMA_INFERENCE_ALPN`]
/// connections and dispatching them to an [`IrohHandler`].
pub struct IrohProvider {
    router: Router,
}

impl IrohProvider {
    /// Wrap an already-built [`Endpoint`] (which must have been built with
    /// [`SOMA_INFERENCE_ALPN`] in its `alpns`) in an accept loop.
    pub fn serve(endpoint: Endpoint, handler: Arc<dyn IrohHandler>) -> Self {
        let proto = InferenceProtocol { handler };
        let router = Router::builder(endpoint).accept(SOMA_INFERENCE_ALPN, proto).spawn();
        IrohProvider { router }
    }

    /// Serve an [`axum::Router`] (the provider's real HTTP stack) over iroh.
    /// Every request is bridged through the router, so auth/voucher/backend
    /// logic is reused verbatim. See [`axum_bridge`].
    pub fn serve_axum(endpoint: Endpoint, router: axum::Router) -> Self {
        Self::serve(endpoint, Arc::new(AxumBridge::new(router)))
    }

    /// Bind a loopback-only endpoint (no relay, no discovery) and serve it.
    /// Suitable for tests and same-host deployments where the buyer is given
    /// the provider's [`EndpointAddr`] directly via [`Self::endpoint_addr`].
    pub async fn bind_local(handler: Arc<dyn IrohHandler>) -> anyhow::Result<Self> {
        let endpoint = build_local_endpoint(true).await?;
        Ok(Self::serve(endpoint, handler))
    }

    /// Like [`Self::bind_local`] but serving an [`axum::Router`].
    pub async fn bind_local_axum(router: axum::Router) -> anyhow::Result<Self> {
        Self::bind_local(Arc::new(AxumBridge::new(router))).await
    }

    /// This provider's public-key identity.
    pub fn endpoint_id(&self) -> EndpointId {
        self.router.endpoint().id()
    }

    /// The local socket addresses the iroh endpoint is bound to. Used by
    /// tests to wire direct-address dialing without n0 discovery.
    pub fn bound_sockets(&self) -> Vec<std::net::SocketAddr> {
        self.router.endpoint().bound_sockets()
    }

    /// A directly-dialable address (id + bound socket addresses) for this
    /// provider. In production the buyer would instead resolve the bare
    /// [`EndpointId`] through discovery; for tests / same-host we hand over
    /// the direct addresses so no relay or DNS is needed.
    pub fn endpoint_addr(&self) -> EndpointAddr {
        let endpoint = self.router.endpoint();
        let mut addr = EndpointAddr::new(endpoint.id());
        for socket in endpoint.bound_sockets() {
            addr = addr.with_ip_addr(socket);
        }
        addr
    }

    /// Stop accepting and shut the endpoint down.
    pub async fn shutdown(self) {
        let _ = self.router.shutdown().await;
    }
}

#[derive(Clone)]
struct InferenceProtocol {
    handler: Arc<dyn IrohHandler>,
}

impl std::fmt::Debug for InferenceProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceProtocol").finish_non_exhaustive()
    }
}

impl ProtocolHandler for InferenceProtocol {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        // Buyers pool one connection and open a fresh bi-stream per request
        // (QUIC multiplexes), so accept streams in a loop and serve each
        // concurrently. The loop ends when the buyer closes the connection.
        let peer = connection.remote_id();
        loop {
            let (send, recv) = match connection.accept_bi().await {
                Ok(streams) => streams,
                Err(_) => break, // connection closed by the buyer / idle timeout
            };
            let handler = self.handler.clone();
            tokio::spawn(async move {
                if let Err(e) = serve_stream(&handler, peer, send, recv).await {
                    tracing::warn!(err = %e, "iroh inference stream failed");
                }
            });
        }
        Ok(())
    }
}

/// Serve a single request stream: read the request, run the handler, stream
/// the response back. A write error (buyer reset the stream) drops `resp.body`,
/// cancelling the provider's in-flight backend work.
async fn serve_stream(
    handler: &Arc<dyn IrohHandler>,
    peer: EndpointId,
    mut send: iroh::endpoint::SendStream,
    mut recv: iroh::endpoint::RecvStream,
) -> anyhow::Result<()> {
    let req: WireRequestMeta = wire::read_meta(&mut recv).await?;
    let body = wire::read_body_to_end(&mut recv, wire::MAX_BODY_LEN).await?;

    let resp = handler.handle(peer, req, body).await;

    let meta = WireResponseMeta { status: resp.status, headers: resp.headers };
    wire::write_meta(&mut send, &meta).await?;

    // Stream the response. If the buyer abandons the request it resets this
    // stream, so `write_all` errors — returning here drops `resp.body`, which
    // tears down the bridge's axum body stream and cancels the provider's
    // in-flight backend work (it stops doing/charging for abandoned work).
    let mut resp_body = resp.body;
    while let Some(chunk) = resp_body.next().await {
        let chunk = chunk?;
        if !chunk.is_empty() {
            send.write_all(&chunk).await.map_err(io_err)?;
        }
    }
    // The connection (owned by the accept loop) outlives this stream and
    // delivers the finished stream's buffered data, so dropping `send` after
    // finish() is safe.
    send.finish().map_err(io_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Buyer (client) side
// ---------------------------------------------------------------------------

/// The buyer/proxy side of the transport. Holds one long-lived [`Endpoint`]
/// and a pool of one [`Connection`] per provider [`EndpointId`]; each request
/// opens a fresh bi-stream on the pooled connection (QUIC multiplexes
/// streams, so this avoids a connect handshake per request).
pub struct IrohBuyer {
    endpoint: Endpoint,
    conns: tokio::sync::Mutex<std::collections::HashMap<EndpointId, Connection>>,
}

impl IrohBuyer {
    /// Wrap an existing endpoint.
    pub fn new(endpoint: Endpoint) -> Self {
        IrohBuyer { endpoint, conns: tokio::sync::Mutex::new(std::collections::HashMap::new()) }
    }

    /// Bind a loopback-only buyer endpoint (no relay, no discovery).
    pub async fn bind_local() -> anyhow::Result<Self> {
        Ok(Self::new(build_local_endpoint(false).await?))
    }

    /// This buyer's public-key identity (what the provider sees as `peer`).
    pub fn endpoint_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    /// Open a bi-stream to `addr`, reusing a pooled connection when one is
    /// live, else dialing a fresh one. A pooled connection whose `open_bi`
    /// fails (idle-timed-out / reset by the peer) is evicted and re-dialed.
    async fn open_bi(
        &self,
        addr: EndpointAddr,
    ) -> anyhow::Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        let id = addr.id;
        let cached = self.conns.lock().await.get(&id).cloned();
        if let Some(conn) = cached {
            if let Ok(streams) = conn.open_bi().await {
                return Ok(streams);
            }
            // Stale connection — drop it and dial afresh below.
            self.conns.lock().await.remove(&id);
        }
        let conn = self.endpoint.connect(addr, SOMA_INFERENCE_ALPN).await?;
        self.conns.lock().await.insert(id, conn.clone());
        Ok(conn.open_bi().await?)
    }

    /// Send a request to `provider` and return a streamed response.
    ///
    /// `provider` is anything convertible into an [`EndpointAddr`] — a bare
    /// [`EndpointId`] (resolved via discovery) or a full address with direct
    /// socket addresses (tests / same-host).
    pub async fn post(
        &self,
        provider: impl Into<EndpointAddr>,
        req: WireRequestMeta,
        body: Bytes,
    ) -> anyhow::Result<BuyerResponse> {
        let (mut send, mut recv) = self.open_bi(provider.into()).await?;

        wire::write_meta(&mut send, &req).await?;
        if !body.is_empty() {
            send.write_all(&body).await.map_err(io_err)?;
        }
        send.finish()?;

        let meta: WireResponseMeta = wire::read_meta(&mut recv).await?;

        // Stream the body back over an mpsc channel. Only the per-request
        // RecvStream is moved into the reader task — the Connection lives in
        // the pool, so it survives for the next request's bi-stream.
        let (tx, rx) = mpsc::channel::<io::Result<Bytes>>(64);
        tokio::spawn(async move {
            loop {
                match recv.read_chunk(64 * 1024).await {
                    Ok(Some(chunk)) => {
                        if tx.send(Ok(chunk.bytes)).await.is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx.send(Err(io_err(e))).await;
                        break;
                    }
                }
            }
        });

        Ok(BuyerResponse {
            status: meta.status,
            headers: meta.headers,
            body: ReceiverStream::new(rx),
        })
    }
}

/// A response received by the buyer. `body` streams the raw response bytes
/// (forward verbatim for SSE; collect via [`BuyerResponse::read_to_end`] for
/// non-streaming).
pub struct BuyerResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: ReceiverStream<io::Result<Bytes>>,
}

impl BuyerResponse {
    /// Collect the entire body into one buffer.
    pub async fn read_to_end(self) -> io::Result<Bytes> {
        let mut buf = Vec::new();
        let mut body = self.body;
        while let Some(item) = body.next().await {
            buf.extend_from_slice(&item?);
        }
        Ok(Bytes::from(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in provider handler: echoes the request path and returns an
    /// OpenAI-shaped completion; emits SSE chunks when the body has
    /// `"stream": true`. Also records the peer id it observed.
    struct ChatHandler {
        seen_peer: Arc<tokio::sync::Mutex<Option<EndpointId>>>,
    }

    #[async_trait::async_trait]
    impl IrohHandler for ChatHandler {
        async fn handle(
            &self,
            peer: EndpointId,
            req: WireRequestMeta,
            body: Bytes,
        ) -> WireResponse {
            *self.seen_peer.lock().await = Some(peer);
            let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
            let is_stream = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

            if is_stream {
                let chunks: Vec<io::Result<Bytes>> = vec![
                    Ok(Bytes::from_static(
                        b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
                    )),
                    Ok(Bytes::from_static(
                        b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
                    )),
                    Ok(Bytes::from_static(b"data: [DONE]\n\n")),
                ];
                WireResponse::stream(
                    200,
                    vec![("content-type".to_string(), "text/event-stream".to_string())],
                    stream::iter(chunks).boxed(),
                )
            } else {
                let resp = serde_json::json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": v.get("model").cloned().unwrap_or(serde_json::Value::Null),
                    "echo_path": req.path,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello world"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
                });
                WireResponse::json(200, &resp)
            }
        }
    }

    fn chat_request(model: &str, stream: bool) -> (WireRequestMeta, Bytes) {
        let meta = WireRequestMeta {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            headers: vec![("authorization".to_string(), "SomaPay v2 …".to_string())],
        };
        let body = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": model,
                "stream": stream,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .unwrap(),
        );
        (meta, body)
    }

    #[test]
    fn secret_persists_and_reloads_to_same_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("provider").join("iroh.key");
        // First call generates + persists; second reloads from disk.
        let first = load_or_create_secret(&path).unwrap();
        assert!(path.exists(), "key file should be written");
        let second = load_or_create_secret(&path).unwrap();
        // Same secret ⇒ same EndpointId across "restarts".
        assert_eq!(first.public(), second.public());
    }

    /// A minimal axum app standing in for the provider's real
    /// `server::handler::build_router` — same shape (`POST
    /// /v1/chat/completions`, JSON or SSE), so it exercises the bridge the
    /// way the production router will be exercised.
    fn chat_app() -> axum::Router {
        use axum::routing::post;

        async fn chat(
            peer: Option<axum::Extension<PeerEndpointId>>,
            body: Bytes,
        ) -> axum::response::Response {
            use axum::response::IntoResponse;
            let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
            let is_stream = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
            // Echo back whether the bridge handed us the authenticated peer,
            // so the test can assert the extension is threaded through axum.
            let saw_peer = peer.is_some();
            if is_stream {
                axum::response::Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(axum::body::Body::from(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n\
                         data: [DONE]\n\n",
                    ))
                    .unwrap()
            } else {
                (
                    axum::http::StatusCode::OK,
                    axum::Json(serde_json::json!({
                        "object": "chat.completion",
                        "model": v.get("model").cloned().unwrap_or(serde_json::Value::Null),
                        "saw_peer": saw_peer,
                    })),
                )
                    .into_response()
            }
        }

        axum::Router::new().route("/v1/chat/completions", post(chat))
    }

    /// End-to-end through the **axum bridge**: a buyer reaches a real
    /// `axum::Router` entirely over iroh. This is how the production provider
    /// stack (auth + backend) will be served.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn buyer_reaches_axum_router_over_iroh() {
        let provider = IrohProvider::bind_local_axum(chat_app()).await.expect("provider");
        let buyer = IrohBuyer::bind_local().await.expect("buyer");

        let (meta, body) = chat_request("axum-model", false);
        let resp = buyer.post(provider.endpoint_addr(), meta, body).await.expect("post");
        assert_eq!(resp.status, 200);

        let bytes = resp.read_to_end().await.expect("read body");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["model"], "axum-model");
        // The bridge threaded the authenticated peer id into the axum request.
        assert_eq!(json["saw_peer"], true);

        provider.shutdown().await;
    }

    /// Streaming SSE through the axum bridge arrives in order over iroh.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn buyer_streams_from_axum_router_over_iroh() {
        let provider = IrohProvider::bind_local_axum(chat_app()).await.expect("provider");
        let buyer = IrohBuyer::bind_local().await.expect("buyer");

        let (meta, body) = chat_request("axum-model", true);
        let resp = buyer.post(provider.endpoint_addr(), meta, body).await.expect("post");
        assert_eq!(resp.status, 200);
        assert!(
            resp.headers.iter().any(|(k, v)| k == "content-type" && v == "text/event-stream"),
            "expected SSE content-type, got {:?}",
            resp.headers
        );

        let text =
            String::from_utf8(resp.read_to_end().await.expect("body").to_vec()).expect("utf8");
        assert!(text.contains("Hi"), "missing delta: {text:?}");
        assert!(text.trim_end().ends_with("[DONE]"), "missing terminator: {text:?}");

        provider.shutdown().await;
    }

    /// Multiple requests on one buyer reuse the pooled connection (QUIC
    /// multiplexes the bi-streams) and the provider's accept loop serves each.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pooled_connection_serves_multiple_requests() {
        let provider = IrohProvider::bind_local_axum(chat_app()).await.expect("provider");
        let buyer = IrohBuyer::bind_local().await.expect("buyer");
        let addr = provider.endpoint_addr();

        for i in 0..3 {
            let (meta, body) = chat_request("axum-model", false);
            let resp = buyer.post(addr.clone(), meta, body).await.expect("post");
            assert_eq!(resp.status, 200, "request {i} must succeed");
            let bytes = resp.read_to_end().await.expect("body");
            let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
            assert_eq!(json["object"], "chat.completion", "request {i}");
        }

        provider.shutdown().await;
    }

    /// End-to-end: a buyer dials a provider purely over iroh (loopback, no
    /// relay), sends a non-streaming chat request, and gets the completion
    /// back. Also asserts the provider authenticated the buyer by its key.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn buyer_connects_to_provider_non_streaming() {
        let seen_peer = Arc::new(tokio::sync::Mutex::new(None));
        let handler = Arc::new(ChatHandler { seen_peer: seen_peer.clone() });
        let provider = IrohProvider::bind_local(handler).await.expect("provider");

        let buyer = IrohBuyer::bind_local().await.expect("buyer");
        let buyer_id = buyer.endpoint_id();

        let (meta, body) = chat_request("test-model", false);
        let resp = buyer.post(provider.endpoint_addr(), meta, body).await.expect("post");
        assert_eq!(resp.status, 200);

        let bytes = resp.read_to_end().await.expect("read body");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["echo_path"], "/v1/chat/completions");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello world");

        // The provider saw the buyer's authenticated public key — the hook an
        // allowlist / channel-payer check would use.
        assert_eq!(*seen_peer.lock().await, Some(buyer_id));

        provider.shutdown().await;
    }

    /// End-to-end streaming: the SSE chunks arrive over iroh in order and
    /// terminate with `[DONE]`, ready to be forwarded to the local agent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn buyer_connects_to_provider_streaming() {
        let seen_peer = Arc::new(tokio::sync::Mutex::new(None));
        let handler = Arc::new(ChatHandler { seen_peer: seen_peer.clone() });
        let provider = IrohProvider::bind_local(handler).await.expect("provider");

        let buyer = IrohBuyer::bind_local().await.expect("buyer");

        let (meta, body) = chat_request("test-model", true);
        let resp = buyer.post(provider.endpoint_addr(), meta, body).await.expect("post");
        assert_eq!(resp.status, 200);
        assert!(
            resp.headers.iter().any(|(k, v)| k == "content-type" && v == "text/event-stream"),
            "expected SSE content-type, got {:?}",
            resp.headers
        );

        let bytes = resp.read_to_end().await.expect("read body");
        let text = String::from_utf8(bytes.to_vec()).expect("utf8");
        // Order preserved end to end.
        let hello = text.find("Hello").expect("first chunk");
        let world = text.find(" world").expect("second chunk");
        let done = text.find("[DONE]").expect("terminator");
        assert!(hello < world && world < done, "SSE chunks out of order: {text:?}");

        provider.shutdown().await;
    }
}
