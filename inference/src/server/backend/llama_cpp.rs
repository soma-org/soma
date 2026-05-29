// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Local llama.cpp backend.
//!
//! Fronts a `llama-server` process (llama.cpp's OpenAI-compatible HTTP
//! server) running on the same machine, serving the Gemma 4 weights every
//! Soma provider hosts. Unlike the OpenRouter / Vast backends there's no
//! upstream API key — the model runs locally — and because the server is
//! ours we can read its real-time load from `/metrics` and feed that back
//! into on-chain pricing.
//!
//! Launch `llama-server` with `--metrics` (and a matching `--parallel N`)
//! for [`saturation`](Backend::saturation) to work; without it the metric
//! reports `None` and the price loop holds steady.

use std::sync::Arc;

use anyhow::Context as _;
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use http::HeaderMap;
use reqwest::Client;

use crate::catalog::{ModelCard, ModelsResponse};
use crate::http_util::{ensure_stream_options_include_usage, pass_outbound};
use crate::openai::ChatRequest;
use crate::server::Config;
use crate::server::backend::{Backend, catalog_from_offerings};

/// Default address `llama-server` binds to when the desktop app launches
/// it as a sidecar.
const DEFAULT_UPSTREAM: &str = "http://127.0.0.1:8080";

pub struct LlamaCppBackend {
    client: Client,
    /// Base URL of the local `llama-server` (no trailing slash, no `/v1`).
    upstream_url: String,
    catalog: Vec<ModelCard>,
    /// Parallel decode slots the server was launched with — the
    /// denominator for the saturation fraction.
    slots: f64,
}

impl LlamaCppBackend {
    pub fn new(cfg: &Config) -> anyhow::Result<Arc<Self>> {
        let upstream_url = cfg
            .backend
            .upstream_url
            .clone()
            .unwrap_or_else(|| DEFAULT_UPSTREAM.to_string())
            .trim_end_matches('/')
            .to_string();
        let slots = cfg.backend.slots.unwrap_or(1).max(1) as f64;
        let catalog = catalog_from_offerings(&cfg.offerings, &cfg.server.public_endpoint);
        // Local inference: a single request can take a while, so allow a
        // generous read timeout. Streaming responses bypass it per-chunk.
        let client = Client::builder().timeout(std::time::Duration::from_secs(300)).build()?;
        Ok(Arc::new(Self { client, upstream_url, catalog, slots }))
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.upstream_url)
    }
}

#[async_trait]
impl Backend for LlamaCppBackend {
    async fn list_models(&self) -> anyhow::Result<ModelsResponse> {
        Ok(ModelsResponse { data: self.catalog.clone() })
    }

    async fn health(&self) -> bool {
        // llama-server returns 200 once weights are loaded, 503 while
        // still loading. Either way a response means the process is up.
        let url = format!("{}/health", self.upstream_url);
        match self.client.get(url).send().await {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }

    async fn saturation(&self) -> Option<f64> {
        let url = format!("{}/metrics", self.upstream_url);
        let body = self.client.get(url).send().await.ok()?.text().await.ok()?;
        // `requests_processing` = slots actively decoding; `requests_deferred`
        // = requests queued waiting for a free slot. Their sum over the slot
        // count is the cleanest "how close to max throughput" signal.
        let processing = parse_prometheus_gauge(&body, "llamacpp:requests_processing")?;
        let deferred = parse_prometheus_gauge(&body, "llamacpp:requests_deferred").unwrap_or(0.0);
        Some(((processing + deferred) / self.slots).clamp(0.0, 1.0))
    }

    async fn chat_completions_stream(
        &self,
        mut req: ChatRequest,
        headers: HeaderMap,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<Bytes>>> {
        ensure_stream_options_include_usage(&mut req);
        let mut h = pass_outbound(&headers);
        h.insert(http::header::ACCEPT, "text/event-stream".parse().unwrap());
        h.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        let body = serde_json::to_vec(&req)?;
        let resp = self.client.post(self.chat_url()).headers(h).body(body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("llama.cpp status {status}: {body}");
        }
        Ok(resp.bytes_stream().map(|r| r.map_err(anyhow::Error::from)).boxed())
    }

    async fn chat_completions(
        &self,
        req: ChatRequest,
        headers: HeaderMap,
    ) -> anyhow::Result<serde_json::Value> {
        let mut h = pass_outbound(&headers);
        h.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        let resp = self
            .client
            .post(self.chat_url())
            .headers(h)
            .json(&req)
            .send()
            .await
            .context("llama.cpp chat completion")?;
        let status = resp.status();
        let v: serde_json::Value = resp.json().await?;
        if !status.is_success() {
            anyhow::bail!("llama.cpp status {status}: {v}");
        }
        Ok(v)
    }
}

/// Read a single Prometheus gauge value out of `llama-server`'s `/metrics`
/// text exposition. Matches a line `name value` or `name{labels} value`,
/// skipping `# HELP`/`# TYPE` comment lines. Returns the first match.
fn parse_prometheus_gauge(body: &str, name: &str) -> Option<f64> {
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(rest) = line.strip_prefix(name) else {
            continue;
        };
        // The char right after the metric name must be a label brace or
        // whitespace — otherwise we matched a longer metric that merely
        // shares this prefix (e.g. `_total`).
        let rest = match rest.chars().next() {
            Some('{') => match rest.split_once('}') {
                Some((_, r)) => r,
                None => continue,
            },
            Some(c) if c.is_whitespace() => rest,
            _ => continue,
        };
        if let Some(tok) = rest.split_whitespace().next() {
            if let Ok(v) = tok.parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 3
# HELP llamacpp:requests_deferred Number of requests deferred.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 5
# TYPE llamacpp:kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio 0.42
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 12345
";

    #[test]
    fn parses_gauges() {
        assert_eq!(parse_prometheus_gauge(SAMPLE, "llamacpp:requests_processing"), Some(3.0));
        assert_eq!(parse_prometheus_gauge(SAMPLE, "llamacpp:requests_deferred"), Some(5.0));
        assert_eq!(parse_prometheus_gauge(SAMPLE, "llamacpp:kv_cache_usage_ratio"), Some(0.42));
    }

    #[test]
    fn no_prefix_false_match() {
        // `n_decode` must not match the longer `n_decode_total`.
        assert_eq!(parse_prometheus_gauge(SAMPLE, "llamacpp:n_decode"), None);
    }

    #[test]
    fn missing_metric_is_none() {
        assert_eq!(parse_prometheus_gauge(SAMPLE, "llamacpp:nonexistent"), None);
    }

    #[test]
    fn parses_labeled_gauge() {
        let body = "llamacpp:requests_processing{model=\"gemma\"} 2\n";
        assert_eq!(parse_prometheus_gauge(body, "llamacpp:requests_processing"), Some(2.0));
    }
}
