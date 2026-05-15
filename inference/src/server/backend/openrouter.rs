// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

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
use crate::server::backend::{catalog_from_offerings, Backend};
use crate::server::Config;

pub struct OpenRouterBackend {
    client: Client,
    api_key: String,
    upstream_url: String,
    catalog: Vec<ModelCard>,
}

impl OpenRouterBackend {
    pub fn new(cfg: &Config) -> anyhow::Result<Arc<Self>> {
        let env_key = cfg
            .backend
            .api_key_env
            .clone()
            .unwrap_or_else(|| "OPENROUTER_API_KEY".to_string());
        let api_key = std::env::var(&env_key)
            .with_context(|| format!("env var {env_key} must be set for OpenRouter backend"))?;
        let upstream_url = cfg
            .backend
            .upstream_url
            .clone()
            .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string());
        let catalog = catalog_from_offerings(&cfg.offerings, &cfg.server.public_endpoint);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;
        Ok(Arc::new(Self { client, api_key, upstream_url, catalog }))
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }
}

#[async_trait]
impl Backend for OpenRouterBackend {
    async fn list_models(&self) -> anyhow::Result<ModelsResponse> {
        Ok(ModelsResponse { data: self.catalog.clone() })
    }

    async fn health(&self) -> bool {
        let url = format!("{}/models", self.upstream_url.trim_end_matches('/'));
        match self.client.get(url).send().await {
            Ok(r) => r.status().as_u16() < 500,
            Err(_) => false,
        }
    }

    async fn chat_completions_stream(
        &self,
        mut req: ChatRequest,
        headers: HeaderMap,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<Bytes>>> {
        ensure_stream_options_include_usage(&mut req);
        let url = format!("{}/chat/completions", self.upstream_url.trim_end_matches('/'));
        let mut h = pass_outbound(&headers);
        h.insert("HTTP-Referer", "https://soma.network".parse().unwrap());
        h.insert("X-Title", "soma-inference".parse().unwrap());
        h.insert(http::header::AUTHORIZATION, self.auth_header().parse().unwrap());
        h.insert(http::header::ACCEPT, "text/event-stream".parse().unwrap());
        h.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        let body = serde_json::to_vec(&req)?;
        let resp = self.client.post(url).headers(h).body(body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("openrouter status {status}: {body}");
        }
        Ok(resp.bytes_stream().map(|r| r.map_err(anyhow::Error::from)).boxed())
    }

    async fn chat_completions(
        &self,
        req: ChatRequest,
        headers: HeaderMap,
    ) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}/chat/completions", self.upstream_url.trim_end_matches('/'));
        let mut h = pass_outbound(&headers);
        h.insert("HTTP-Referer", "https://soma.network".parse().unwrap());
        h.insert("X-Title", "soma-inference".parse().unwrap());
        h.insert(http::header::AUTHORIZATION, self.auth_header().parse().unwrap());
        h.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        let resp = self.client.post(url).headers(h).json(&req).send().await?;
        let status = resp.status();
        let v: serde_json::Value = resp.json().await?;
        if !status.is_success() {
            anyhow::bail!("openrouter status {status}: {v}");
        }
        Ok(v)
    }
}
