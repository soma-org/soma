// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Indexer-backed [`ProviderRegistry`].
//!
//! `list_providers` queries the soma-graphql `providers(activeWithinMs:)`
//! endpoint, which sources from the `soma_providers` indexer table —
//! itself populated from `RegisterProvider` / `UpdateProvider`
//! transactions on chain. `register_provider` is a no-op: registration
//! happens directly through `sdk::provider::register_or_update` at
//! server boot, so the indexer picks it up at the next checkpoint.
//!
//! Stale providers are filtered by `active_within_ms` against
//! `Provider.registered_at_ms` (heartbeat-based liveness).

use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use types::base::SomaAddress;

use crate::chain::types::*;
use crate::chain::ProviderRegistry;

/// Default staleness threshold — providers whose last on-chain
/// heartbeat is older than this are dropped from the active set.
/// Servers heartbeat via `UpdateProvider` every ~10 minutes (see
/// `soma inference serve --heartbeat-interval-secs`), so 30 minutes
/// tolerates a missed beat without dropping live providers.
pub const DEFAULT_ACTIVE_WINDOW_MS: u64 = 30 * 60 * 1000;

/// HTTP client over the soma-graphql `providers` query.
#[derive(Clone)]
pub struct IndexerProviderRegistry {
    http: reqwest::Client,
    url: String,
    active_window_ms: u64,
}

impl IndexerProviderRegistry {
    pub fn new(url: String) -> Self {
        Self::with_window(url, DEFAULT_ACTIVE_WINDOW_MS)
    }

    pub fn with_window(url: String, active_window_ms: u64) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("reqwest client");
        Self { http, url, active_window_ms }
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

#[async_trait]
impl ProviderRegistry for IndexerProviderRegistry {
    async fn list_providers(&self) -> Result<Vec<ProviderRecord>, ChainError> {
        let query = r#"
            query Providers($w: Int!) {
                providers(first: 50, activeWithinMs: $w) {
                    edges { node { address endpoint registeredAtMs } }
                }
            }
        "#;
        // Clamp to i32 — the Int scalar's domain. 30 minutes (the
        // default window) is well within i32 range; any caller that
        // sets a larger window should switch the GraphQL field to
        // BigInt at the same time.
        let window_i32: i32 =
            (self.active_window_ms as i64).clamp(0, i32::MAX as i64) as i32;
        let body = serde_json::json!({
            "query": query,
            "variables": { "w": window_i32 },
        });
        let resp = self
            .http
            .post(&self.url)
            .json(&body)
            .send()
            .await
            .map_err(|e| ChainError::Rpc(format!("graphql post: {e}")))?;
        if !resp.status().is_success() {
            return Err(ChainError::Rpc(format!(
                "indexer returned status {}",
                resp.status()
            )));
        }
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| ChainError::Rpc(format!("graphql json: {e}")))?;
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
            let last_heartbeat_ms: u64 = parsed.registered_at_ms.parse().unwrap_or(0);
            out.push(ProviderRecord {
                address,
                pubkey_hex: String::new(),
                endpoint: parsed.endpoint,
                last_heartbeat_ms,
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
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GqlProvider {
    address: String,
    endpoint: String,
    /// `BigInt` scalars come back as strings.
    registered_at_ms: String,
}
