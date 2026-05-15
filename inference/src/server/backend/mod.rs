// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Provider backends — adapters in front of OpenAI-compatible upstreams.
//!
//! Implement [`Backend`] in a new file under this module to plug in a new
//! upstream. The OpenRouter and Vast backends are about ~120 lines each and
//! make good references.

pub mod openrouter;
pub mod vast;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use http::HeaderMap;

use crate::catalog::{ModelCard, ModelsResponse, SomaInfo};
use crate::openai::ChatRequest;

#[async_trait]
pub trait Backend: Send + Sync + 'static {
    async fn list_models(&self) -> anyhow::Result<ModelsResponse>;
    async fn health(&self) -> bool;

    async fn chat_completions_stream(
        &self,
        req: ChatRequest,
        headers: HeaderMap,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<Bytes>>>;

    async fn chat_completions(
        &self,
        req: ChatRequest,
        headers: HeaderMap,
    ) -> anyhow::Result<serde_json::Value>;
}

pub(crate) fn catalog_from_offerings(
    offerings: &[ModelCard],
    public_endpoint: &str,
) -> Vec<ModelCard> {
    offerings
        .iter()
        .cloned()
        .map(|mut c| {
            c.soma = Some(SomaInfo {
                provider_address: String::new(),
                endpoint: public_endpoint.to_string(),
            });
            c
        })
        .collect()
}

pub fn fill_soma_info(catalog: &mut [ModelCard], provider_address: &str) {
    for c in catalog {
        if let Some(s) = c.soma.as_mut() {
            s.provider_address = provider_address.to_string();
        }
    }
}
