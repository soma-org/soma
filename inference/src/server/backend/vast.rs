// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Vast.ai backend. Two ways to point at vLLM:
//! - Serverless: `[backend].endpoint_name` → `https://openai.vast.ai/<name>`.
//! - Direct rental: `[backend].upstream_url` → `http://<public_ip>:<port>`.
//!
//! Either way, `VAST_API_KEY` is required (vLLM ignores it unless launched
//! with `--api-key`, but the constructor needs *something* in the
//! Authorization header).

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

pub struct VastBackend {
    client: Client,
    api_key: String,
    upstream_url: String,
    catalog: Vec<ModelCard>,
}

impl VastBackend {
    pub fn new(cfg: &Config) -> anyhow::Result<Arc<Self>> {
        let env_key = cfg
            .backend
            .api_key_env
            .clone()
            .unwrap_or_else(|| "VAST_API_KEY".to_string());
        let api_key = std::env::var(&env_key)
            .with_context(|| format!("env var {env_key} must be set for Vast backend"))?;

        let upstream_url = match (
            cfg.backend.upstream_url.as_ref(),
            cfg.backend.endpoint_name.as_ref(),
        ) {
            (Some(u), _) => u.trim_end_matches('/').to_string(),
            (None, Some(name)) => {
                let name = name.trim().trim_matches('/');
                if name.is_empty() {
                    anyhow::bail!("vast backend: endpoint_name must not be empty");
                }
                format!("https://openai.vast.ai/{name}")
            }
            (None, None) => anyhow::bail!(
                "vast backend: set either [backend].upstream_url or [backend].endpoint_name"
            ),
        };

        let catalog = catalog_from_offerings(&cfg.offerings, &cfg.server.public_endpoint);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
        Ok(Arc::new(Self { client, api_key, upstream_url, catalog }))
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }
}

#[async_trait]
impl Backend for VastBackend {
    async fn list_models(&self) -> anyhow::Result<ModelsResponse> {
        Ok(ModelsResponse { data: self.catalog.clone() })
    }

    async fn health(&self) -> bool {
        let url = format!("{}/v1/models", self.upstream_url);
        match self
            .client
            .get(url)
            .header(http::header::AUTHORIZATION, self.auth_header())
            .send()
            .await
        {
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
        let url = format!("{}/v1/chat/completions", self.upstream_url);
        let mut h = pass_outbound(&headers);
        h.insert(http::header::AUTHORIZATION, self.auth_header().parse().unwrap());
        h.insert(http::header::ACCEPT, "text/event-stream".parse().unwrap());
        h.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        let body = serde_json::to_vec(&req)?;
        let resp = self.client.post(url).headers(h).body(body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("vast status {status}: {body}");
        }
        Ok(resp.bytes_stream().map(|r| r.map_err(anyhow::Error::from)).boxed())
    }

    async fn chat_completions(
        &self,
        req: ChatRequest,
        headers: HeaderMap,
    ) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}/v1/chat/completions", self.upstream_url);
        let mut h = pass_outbound(&headers);
        h.insert(http::header::AUTHORIZATION, self.auth_header().parse().unwrap());
        h.insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        let resp = self.client.post(url).headers(h).json(&req).send().await?;
        let status = resp.status();
        let v: serde_json::Value = resp.json().await?;
        if !status.is_success() {
            anyhow::bail!("vast status {status}: {v}");
        }
        Ok(v)
    }
}
