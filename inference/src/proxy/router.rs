// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use reqwest::Client;
use tokio::sync::RwLock;
use types::base::SomaAddress;

use crate::catalog::{ModelCard, ModelsResponse};
use crate::chain::types::{ChannelState, OpenChannelParams, ProviderRecord};
use crate::chain::Discovery;
use crate::now_ms;
use crate::pricing;
use crate::proxy::config::Config;
use crate::proxy::state::{ChannelSlot, ClientStore};

#[derive(Clone)]
pub struct ProviderInfo {
    pub address: SomaAddress,
    pub pubkey_hex: String,
    pub endpoint: String,
    pub catalog: Vec<ModelCard>,
}

pub struct Router {
    pub chain: Arc<dyn Discovery>,
    pub http: Client,
    pub store: ClientStore,
    pub cfg: Arc<Config>,
    pub client_address: SomaAddress,
    pub client_pubkey_hex: String,
    cache: Arc<RwLock<CacheState>>,
}

struct CacheState {
    last_refresh: Option<Instant>,
    providers: Vec<ProviderInfo>,
}

impl Router {
    pub fn new(
        chain: Arc<dyn Discovery>,
        store: ClientStore,
        cfg: Arc<Config>,
        client_address: SomaAddress,
        client_pubkey_hex: String,
    ) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("build http client");
        Self {
            chain,
            http,
            store,
            cfg,
            client_address,
            client_pubkey_hex,
            cache: Arc::new(RwLock::new(CacheState {
                last_refresh: None,
                providers: Vec::new(),
            })),
        }
    }

    pub async fn refresh_providers(&self) -> anyhow::Result<()> {
        let recs: Vec<ProviderRecord> = self.chain.list_providers().await?;
        let mut providers = Vec::new();
        for rec in recs {
            match self.fetch_provider_info(&rec.endpoint).await {
                Ok(info) => providers.push(info),
                Err(e) => tracing::warn!(addr = %rec.address, err = %e, "provider unreachable"),
            }
        }
        let mut g = self.cache.write().await;
        g.providers = providers;
        g.last_refresh = Some(Instant::now());
        Ok(())
    }

    async fn fetch_provider_info(&self, endpoint: &str) -> anyhow::Result<ProviderInfo> {
        let info_url = format!("{}/soma/info", endpoint.trim_end_matches('/'));
        let info: serde_json::Value = self.http.get(info_url).send().await?.json().await?;
        let pubkey_hex = info
            .get("pubkey_hex")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let address_str = info
            .get("address")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let address = SomaAddress::from_hex_literal(address_str)
            .or_else(|_| SomaAddress::from_hex(address_str))
            .map_err(|e| anyhow::anyhow!("provider /soma/info bad address {address_str}: {e}"))?;

        let models_url = format!("{}/v1/models", endpoint.trim_end_matches('/'));
        let mr: ModelsResponse = self.http.get(models_url).send().await?.json().await?;
        Ok(ProviderInfo {
            address,
            pubkey_hex,
            endpoint: endpoint.to_string(),
            catalog: mr.data,
        })
    }

    pub async fn ensure_cache(&self) -> anyhow::Result<()> {
        let need = {
            let g = self.cache.read().await;
            match g.last_refresh {
                Some(t) => {
                    g.providers.is_empty()
                        || t.elapsed() > Duration::from_secs(self.cfg.discovery.provider_cache_ttl_secs)
                }
                None => true,
            }
        };
        if need {
            self.refresh_providers().await?;
        }
        Ok(())
    }

    pub async fn pick_provider_for_model(
        &self,
        model: &str,
    ) -> anyhow::Result<Option<(ProviderInfo, ModelCard)>> {
        self.ensure_cache().await?;
        let g = self.cache.read().await;
        let mut candidates: Vec<(ProviderInfo, ModelCard)> = Vec::new();
        for p in &g.providers {
            if let Some(c) = p.catalog.iter().find(|c| c.id == model) {
                candidates.push((p.clone(), c.clone()));
            }
        }
        if candidates.is_empty() {
            return Ok(None);
        }
        candidates.sort_by(|a, b| {
            let pa = pricing::parse_fixed(&a.1.pricing.prompt, 12)
                + pricing::parse_fixed(&a.1.pricing.completion, 12);
            let pb = pricing::parse_fixed(&b.1.pricing.prompt, 12)
                + pricing::parse_fixed(&b.1.pricing.completion, 12);
            pa.cmp(&pb)
        });
        Ok(Some(candidates.remove(0)))
    }

    pub async fn aggregate_models(&self) -> HashMap<String, (ProviderInfo, ModelCard)> {
        let _ = self.ensure_cache().await;
        let g = self.cache.read().await;
        let mut out: HashMap<String, (ProviderInfo, ModelCard)> = HashMap::new();
        for p in &g.providers {
            for c in &p.catalog {
                let pa = pricing::parse_fixed(&c.pricing.prompt, 12)
                    + pricing::parse_fixed(&c.pricing.completion, 12);
                if let Some(existing) = out.get(&c.id) {
                    let pe = pricing::parse_fixed(&existing.1.pricing.prompt, 12)
                        + pricing::parse_fixed(&existing.1.pricing.completion, 12);
                    if pa < pe {
                        out.insert(c.id.clone(), (p.clone(), c.clone()));
                    }
                } else {
                    out.insert(c.id.clone(), (p.clone(), c.clone()));
                }
            }
        }
        out
    }

    pub async fn aggregate_health(&self) -> bool {
        let _ = self.ensure_cache().await;
        let g = self.cache.read().await;
        for p in &g.providers {
            let url = format!("{}/health", p.endpoint.trim_end_matches('/'));
            if let Ok(r) = self.http.get(url).send().await {
                if r.status().is_success() {
                    return true;
                }
            }
        }
        false
    }

    pub async fn ensure_channel(
        &self,
        provider: &ProviderInfo,
    ) -> anyhow::Result<Arc<tokio::sync::Mutex<ChannelSlot>>> {
        if let Some(h) = self.store.read_pointer(&provider.address) {
            if let Ok(chan) = self.chain.channel(&h).await {
                if chan.status == crate::chain::types::ChannelStatus::Open {
                    if let Some(slot) = self.store.slot(&h).await {
                        // Reuse if there's still meaningful headroom (>$0.04 of slack).
                        let g = slot.lock().await;
                        if g.state
                            .deposit_micros
                            .saturating_sub(g.state.cumulative_authorized_micros)
                            > 40_000
                        {
                            drop(g);
                            return Ok(slot);
                        }
                    }
                }
            }
        }
        // Open a fresh channel.
        let expires_ms = now_ms() + self.cfg.wallet.channel_expires_secs * 1000;
        let handle = self
            .chain
            .open_channel(OpenChannelParams {
                client: self.client_address,
                client_pubkey_hex: self.client_pubkey_hex.clone(),
                provider: provider.address,
                deposit_micros: self.cfg.wallet.default_deposit_micros,
                expires_ms,
            })
            .await
            .context("open_channel")?;
        let chan: ChannelState = self.chain.channel(&handle).await?;
        let slot = self
            .store
            .init_slot(&chan, &provider.pubkey_hex, &provider.endpoint)
            .await?;
        self.store.write_pointer(&provider.address, &handle)?;
        Ok(slot)
    }
}
