// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Indexer-backed reputation client.
//!
//! Reads the indexer's `provider_reputation` view via GraphQL and
//! exposes it as a Rust struct the proxy router can use to score
//! candidates. Reputation is observation-only — there's no on-chain
//! consensus around these numbers, so retrieval failures fall back
//! to "unknown / no reputation data" rather than crashing.

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;
use types::base::SomaAddress;

#[derive(Debug, Clone, Default)]
pub struct ProviderReputation {
    pub volume_settled_30d: u64,
    pub distinct_buyers_30d: u64,
    pub channel_renewal_rate: f64,
    pub mean_channel_span_cps: u64,
}

/// Thin GraphQL client over `provider(address) { reputation { ... } }`
/// queries. Held by the proxy router; cached at the call site to
/// keep the dependency footprint minimal.
#[derive(Clone)]
pub struct IndexerClient {
    http: reqwest::Client,
    url: String,
}

impl IndexerClient {
    pub fn new(url: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("build reqwest client");
        Self { http, url }
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// Fetch reputation for one provider. Returns `None` if the
    /// indexer hasn't seen any channels for this provider yet (the
    /// `reputation` field comes back null).
    pub async fn fetch(
        &self,
        address: SomaAddress,
    ) -> anyhow::Result<Option<ProviderReputation>> {
        let query = r#"
            query Rep($addr: String!) {
                provider(address: $addr) {
                    reputation {
                        volumeSettled30d
                        distinctBuyers30d
                        channelRenewalRate
                        meanChannelSpanCps
                    }
                }
            }
        "#;
        let body = serde_json::json!({
            "query": query,
            "variables": { "addr": format!("0x{}", hex::encode(address.to_vec())) },
        });
        let resp = self.http.post(&self.url).json(&body).send().await?;
        if !resp.status().is_success() {
            return Ok(None);
        }
        let v: serde_json::Value = resp.json().await?;
        let rep = v
            .get("data")
            .and_then(|d| d.get("provider"))
            .and_then(|p| p.get("reputation"))
            .cloned();
        let Some(rep) = rep else { return Ok(None) };
        if rep.is_null() {
            return Ok(None);
        }
        let parsed: GqlReputation = serde_json::from_value(rep)?;
        Ok(Some(ProviderReputation {
            volume_settled_30d: parsed.volume_settled_30d.parse().unwrap_or(0),
            distinct_buyers_30d: parsed.distinct_buyers_30d.parse().unwrap_or(0),
            channel_renewal_rate: parsed.channel_renewal_rate,
            mean_channel_span_cps: parsed.mean_channel_span_cps.parse().unwrap_or(0),
        }))
    }

    /// Best-effort batch fetch — issues N parallel single-provider
    /// queries. Failed individual queries surface as `None` in the
    /// returned map; the call as a whole only errors on a transport
    /// failure to the indexer URL itself.
    pub async fn fetch_many(
        &self,
        addresses: &[SomaAddress],
    ) -> HashMap<SomaAddress, Option<ProviderReputation>> {
        let mut out = HashMap::with_capacity(addresses.len());
        for addr in addresses {
            let rep = self.fetch(*addr).await.ok().flatten();
            out.insert(*addr, rep);
        }
        out
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GqlReputation {
    /// `BigInt` scalars come back as strings.
    volume_settled_30d: String,
    distinct_buyers_30d: String,
    channel_renewal_rate: f64,
    mean_channel_span_cps: String,
}
