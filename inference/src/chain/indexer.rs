// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Indexer-backed [`ProviderRegistry`].
//!
//! `list_providers` queries the soma-graphql `providers` endpoint,
//! which sources from the `soma_providers` indexer table — itself
//! populated from `RegisterProvider` / `UpdateProvider` /
//! `UnregisterProvider` transactions on chain. `register_provider`
//! is a no-op: registration happens directly through
//! `sdk::provider::register_or_update` at server boot, so the indexer
//! picks it up at the next checkpoint.
//!
//! Liveness is purely off-chain: the proxy probes each returned
//! `endpoint` over HTTP. Stale entries (servers that disappeared
//! without unregistering) are filtered by failed probes, not by any
//! on-chain timestamp.

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use types::base::SomaAddress;

use crate::chain::ProviderRegistry;
use crate::chain::types::*;

/// HTTP client over the soma-graphql `providers` query.
#[derive(Clone)]
pub struct IndexerProviderRegistry {
    http: reqwest::Client,
    url: String,
}

impl IndexerProviderRegistry {
    pub fn new(url: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("reqwest client");
        Self { http, url }
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

#[async_trait]
impl ProviderRegistry for IndexerProviderRegistry {
    async fn list_providers(&self) -> Result<Vec<ProviderRecord>, ChainError> {
        let query = r#"
            query Providers {
                providers(first: 50) {
                    edges { node { address endpoint irohEndpointId } }
                }
            }
        "#;
        let body = serde_json::json!({ "query": query });
        let resp = self
            .http
            .post(&self.url)
            .json(&body)
            .send()
            .await
            .map_err(|e| ChainError::Rpc(format!("graphql post: {e}")))?;
        if !resp.status().is_success() {
            return Err(ChainError::Rpc(format!("indexer returned status {}", resp.status())));
        }
        let body: serde_json::Value =
            resp.json().await.map_err(|e| ChainError::Rpc(format!("graphql json: {e}")))?;
        let edges = body
            .pointer("/data/providers/edges")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(edges.len());
        for edge in edges {
            let Some(node) = edge.get("node") else { continue };
            let parsed: GqlProvider = match serde_json::from_value(node.clone()) {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!(err = %e, "skipping malformed provider edge");
                    continue;
                }
            };
            let addr_hex = parsed.address.strip_prefix("0x").unwrap_or(&parsed.address);
            let addr_bytes = match hex::decode(addr_hex) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let address = match SomaAddress::try_from(addr_bytes.as_slice()) {
                Ok(a) => a,
                Err(_) => continue,
            };
            out.push(ProviderRecord {
                address,
                pubkey_hex: String::new(),
                endpoint: parsed.endpoint,
                iroh_endpoint_id: parsed.iroh_endpoint_id,
                // Production dials by EndpointId via n0 discovery — no direct
                // addresses needed here.
                iroh_direct_addrs: Vec::new(),
            });
        }
        Ok(out)
    }

    async fn register_provider(&self, _record: ProviderRecord) -> Result<(), ChainError> {
        // No-op: on-chain `RegisterProvider` is the source of truth.
        // Server boot calls `sdk::provider::register_or_update`
        // directly; the indexer mirrors the resulting Provider object
        // into `soma_providers` automatically.
        Ok(())
    }

    fn indexer_url(&self) -> Option<&str> {
        Some(&self.url)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GqlProvider {
    address: String,
    endpoint: String,
    #[serde(default)]
    iroh_endpoint_id: String,
}
