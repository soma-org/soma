// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use reqwest::Client;
use tokio::sync::RwLock;
use ::types::base::SomaAddress;
use ::types::object::{CoinType, ObjectID};

use crate::catalog::{ModelCard, ModelsResponse};
use crate::chain::{ChannelSurface, ProviderRegistry, ProviderRecord};
use crate::proxy::config::{Config, RoutingMode, RoutingWeights};
use crate::proxy::state::{ChannelSlot, ClientStore};
use crate::reputation::{IndexerClient, ProviderReputation};

/// Sum of prompt + completion price (per-token, fixed-point µ-USD).
/// Pulled out of [`Router::price_score`] so unit tests can call it
/// without constructing a [`Router`].
fn price_score(card: &ModelCard) -> u128 {
    crate::pricing::parse_fixed(&card.pricing.prompt, 12)
        + crate::pricing::parse_fixed(&card.pricing.completion, 12)
}

/// Combined provider score for `RoutingMode::Weighted`. Higher = more
/// preferred.
///
/// Formula:
/// ```text
///   -w_price·price + w_volume·log(volume_settled_30d + 1)
///                  + w_buyers·log(distinct_buyers_30d + 1)
///                  + w_renewal·channel_renewal_rate
/// ```
///
/// Logs damp high-cardinality signals so a single big provider
/// doesn't dominate. Reputation `None` contributes 0 (unknown
/// provider scores strictly worse than one with positive signal).
/// Pulled out of [`Router::weighted_score`] for unit-testability.
fn weighted_score(
    w: &RoutingWeights,
    card: &ModelCard,
    rep: Option<ProviderReputation>,
) -> f64 {
    let price = price_score(card) as f64;
    let mut score = -w.price * price;
    if let Some(r) = rep {
        score += w.volume * (r.volume_settled_30d as f64 + 1.0).ln();
        score += w.buyers * (r.distinct_buyers_30d as f64 + 1.0).ln();
        score += w.renewal * r.channel_renewal_rate;
    }
    score
}

#[derive(Clone)]
pub struct ProviderInfo {
    pub address: SomaAddress,
    pub pubkey_hex: String,
    pub endpoint: String,
    pub catalog: Vec<ModelCard>,
}

pub struct Router {
    pub registry: Arc<dyn ProviderRegistry>,
    pub chain: Arc<dyn ChannelSurface>,
    pub http: Client,
    pub store: ClientStore,
    pub cfg: Arc<Config>,
    pub client_address: SomaAddress,
    cache: Arc<RwLock<CacheState>>,
    /// Optional reputation client. Populated only when
    /// `cfg.routing.indexer_url` is set; weighted routing falls back
    /// to price-only with a warning when this is `None`.
    indexer: Option<IndexerClient>,
}

struct CacheState {
    last_refresh: Option<Instant>,
    providers: Vec<ProviderInfo>,
}

impl Router {
    pub fn new(
        registry: Arc<dyn ProviderRegistry>,
        chain: Arc<dyn ChannelSurface>,
        store: ClientStore,
        cfg: Arc<Config>,
        client_address: SomaAddress,
    ) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("build http client");
        let indexer = cfg
            .routing
            .indexer_url
            .as_ref()
            .map(|url| IndexerClient::new(url.clone()));
        Self {
            registry,
            chain,
            http,
            store,
            cfg,
            client_address,
            cache: Arc::new(RwLock::new(CacheState {
                last_refresh: None,
                providers: Vec::new(),
            })),
            indexer,
        }
    }

    pub async fn refresh_providers(&self) -> anyhow::Result<()> {
        let recs: Vec<ProviderRecord> = self.registry.list_providers().await?;
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
                        || t.elapsed() > Duration::from_secs(self.cfg.provider_cache_ttl_secs)
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

        match self.cfg.routing.mode {
            RoutingMode::Price => {
                candidates.sort_by(|a, b| {
                    self.price_score(&a.1).cmp(&self.price_score(&b.1))
                });
                Ok(Some(candidates.remove(0)))
            }
            RoutingMode::Weighted => {
                let Some(indexer) = self.indexer.clone() else {
                    tracing::warn!(
                        "routing.mode = weighted but no indexer_url configured; falling back to price"
                    );
                    candidates.sort_by(|a, b| {
                        self.price_score(&a.1).cmp(&self.price_score(&b.1))
                    });
                    return Ok(Some(candidates.remove(0)));
                };
                let addrs: Vec<_> = candidates.iter().map(|(p, _)| p.address).collect();
                let reps = indexer.fetch_many(&addrs).await;
                // Pick the highest scorer. Higher = better (price is
                // already negated inside `weighted_score`).
                candidates.sort_by(|a, b| {
                    let sa = self.weighted_score(&a.1, reps.get(&a.0.address).cloned().flatten());
                    let sb = self.weighted_score(&b.1, reps.get(&b.0.address).cloned().flatten());
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(Some(candidates.remove(0)))
            }
        }
    }

    fn price_score(&self, card: &ModelCard) -> u128 {
        price_score(card)
    }

    fn weighted_score(&self, card: &ModelCard, rep: Option<ProviderReputation>) -> f64 {
        weighted_score(&self.cfg.routing.weights, card, rep)
    }

    pub async fn aggregate_models(&self) -> HashMap<String, (ProviderInfo, ModelCard)> {
        let _ = self.ensure_cache().await;
        let g = self.cache.read().await;
        let mut out: HashMap<String, (ProviderInfo, ModelCard)> = HashMap::new();
        for p in &g.providers {
            for c in &p.catalog {
                let pa = crate::pricing::parse_fixed(&c.pricing.prompt, 12)
                    + crate::pricing::parse_fixed(&c.pricing.completion, 12);
                if let Some(existing) = out.get(&c.id) {
                    let pe = crate::pricing::parse_fixed(&existing.1.pricing.prompt, 12)
                        + crate::pricing::parse_fixed(&existing.1.pricing.completion, 12);
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

    /// Return an open channel to `provider` — reuse an existing one
    /// if it has meaningful headroom, otherwise lazily open a new
    /// on-chain channel. On a cache miss, also asks the provider's
    /// `/soma/channel/{id}` endpoint for the highest cumulative they
    /// hold so the proxy never issues a non-monotonic voucher after a
    /// restart.
    pub async fn ensure_channel(
        &self,
        provider: &ProviderInfo,
    ) -> anyhow::Result<Arc<tokio::sync::Mutex<ChannelSlot>>> {
        if let Some(id) = self.store.read_pointer(&provider.address).await {
            if let Ok(chan) = self.chain.get(id).await {
                if chan.close_requested_at_ms().is_none() {
                    if let Some(slot) = self.store.slot(&id).await {
                        let g = slot.lock().await;
                        if g.state
                            .deposit_micros
                            .saturating_sub(g.state.cumulative_authorized_micros)
                            > 40_000
                        {
                            drop(g);
                            return Ok(slot);
                        }
                    } else {
                        // Pointer exists, slot doesn't — cold-start
                        // path. Hydrate from chain + provider.
                        let cumulative = self
                            .fetch_provider_last_cumulative(&provider.endpoint, id)
                            .await
                            .unwrap_or(chan.settled_amount());
                        let floor = cumulative.max(chan.settled_amount());
                        if chan.deposit().saturating_sub(floor) > 40_000 {
                            let slot = self
                                .store
                                .install_slot(
                                    id,
                                    provider.address,
                                    provider.endpoint.clone(),
                                    chan.deposit(),
                                    floor,
                                )
                                .await;
                            return Ok(slot);
                        }
                    }
                }
            }
        }
        // Lazy on-chain open.
        let id = self
            .chain
            .open(
                provider.address,
                CoinType::Usdc,
                self.cfg.default_deposit_micros,
            )
            .await
            .context("open_channel")?;
        let chan = self.chain.get(id).await.context("get_channel after open")?;
        let slot = self
            .store
            .install_slot(
                id,
                provider.address,
                provider.endpoint.clone(),
                chan.deposit(),
                chan.settled_amount(),
            )
            .await;
        Ok(slot)
    }

    /// Ask the provider's `/soma/channel/{id}` endpoint for the
    /// highest cumulative voucher they hold. Returns `None` if the
    /// provider doesn't expose the endpoint, the channel is unknown
    /// to them, or any HTTP error — callers fall back to the chain's
    /// `settled_amount`.
    async fn fetch_provider_last_cumulative(
        &self,
        endpoint: &str,
        channel_id: ObjectID,
    ) -> Option<u64> {
        let url = format!(
            "{}/soma/channel/{}",
            endpoint.trim_end_matches('/'),
            channel_id
        );
        let resp = self.http.get(url).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let v: serde_json::Value = resp.json().await.ok()?;
        v.get("last_cumulative_micros").and_then(|x| x.as_u64())
    }
}

// ---------------------------------------------------------------------
// Unit tests for the routing arithmetic.
//
// Constructing a full Router is heavy (registry/chain/store/http
// stubs), so these tests target the standalone `price_score` and
// `weighted_score` helpers. The Router methods are 1-line wrappers
// around them — the math is what's worth pinning down.
// ---------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Architecture, ModelCard, Pricing, TopProvider};

    fn card(prompt: &str, completion: &str) -> ModelCard {
        ModelCard {
            id: "m".to_string(),
            name: "m".to_string(),
            canonical_slug: None,
            hugging_face_id: None,
            created: 0,
            description: None,
            context_length: 1024,
            architecture: Architecture {
                input_modalities: vec!["text".into()],
                output_modalities: vec!["text".into()],
                tokenizer: "x".into(),
                instruct_type: None,
            },
            top_provider: TopProvider {
                context_length: 1024,
                max_completion_tokens: None,
                is_moderated: false,
            },
            supported_parameters: vec![],
            default_parameters: None,
            expiration_date: None,
            pricing: Pricing {
                prompt: prompt.to_string(),
                completion: completion.to_string(),
                request: "0".into(),
                image: "0".into(),
                input_cache_read: "0".into(),
                input_cache_write: "0".into(),
            },
            soma: None,
        }
    }

    fn rep(volume: u64, buyers: u64, renewal: f64) -> ProviderReputation {
        ProviderReputation {
            volume_settled_30d: volume,
            distinct_buyers_30d: buyers,
            channel_renewal_rate: renewal,
            mean_channel_span_cps: 0,
        }
    }

    /// `price_score` is monotonic in (prompt + completion).
    #[test]
    fn price_score_is_sum() {
        let cheap = price_score(&card("0.0000001", "0.0000002"));
        let expensive = price_score(&card("0.0000004", "0.0000005"));
        assert!(cheap < expensive, "cheap={cheap} >= expensive={expensive}");
        // 1+2 vs 4+5 in 12-decimal fixed point.
        assert_eq!(expensive, 9 * cheap / 3);
    }

    /// In `Price` mode, the cheaper provider wins.
    #[test]
    fn price_mode_picks_cheaper() {
        let cheap = card("0.0000001", "0.0000002"); // 3 units
        let pricey = card("0.0000010", "0.0000020"); // 30 units
        assert!(price_score(&cheap) < price_score(&pricey));
    }

    /// With a strong reputation tilt, a higher-rep / higher-price
    /// provider beats a low-rep / low-price one.
    ///
    /// `price_score` is in 12-decimal fixed point — `"0.0000001"`
    /// becomes `100_000` units — so weights here look "small" relative
    /// to typical price-weight defaults. The point is the *ratio*:
    /// reputation gain (log-scaled, ~10s of units) must exceed the
    /// price penalty (linear, ~10s–1000s of units) for high-rep
    /// providers to win against pricier competition.
    #[test]
    fn weighted_mode_prefers_high_reputation_when_weights_tilt_that_way() {
        let w = RoutingWeights {
            price: 0.00001,
            volume: 5.0,
            renewal: 1.0,
            buyers: 2.0,
        };

        let cheap_low_rep = card("0.0000001", "0.0000002");
        let cheap_low_rep_rep = Some(rep(0, 0, 0.0));

        let pricey_high_rep = card("0.0000010", "0.0000020");
        let pricey_high_rep_rep = Some(rep(1_000_000, 50, 0.5));

        let s_cheap = weighted_score(&w, &cheap_low_rep, cheap_low_rep_rep);
        let s_pricey = weighted_score(&w, &pricey_high_rep, pricey_high_rep_rep);
        assert!(
            s_pricey > s_cheap,
            "expected high-rep to outscore cheap-no-rep: cheap={s_cheap}, pricey={s_pricey}"
        );
    }

    /// With a strong price tilt, the cheaper provider still wins
    /// even against a higher-rep competitor.
    #[test]
    fn weighted_mode_falls_back_to_price_when_weights_tilt_that_way() {
        let w = RoutingWeights { price: 100.0, volume: 0.01, renewal: 0.01, buyers: 0.01 };

        let cheap = card("0.0000001", "0.0000002");
        let pricey = card("0.0000010", "0.0000020");

        let s_cheap = weighted_score(&w, &cheap, Some(rep(0, 0, 0.0)));
        let s_pricey = weighted_score(&w, &pricey, Some(rep(1_000_000, 100, 1.0)));
        assert!(
            s_cheap > s_pricey,
            "expected cheap to outscore pricey when w_price dominates: cheap={s_cheap}, pricey={s_pricey}"
        );
    }

    /// Reputation `None` contributes zero — an unknown provider
    /// scores strictly less than the same provider with positive
    /// signal.
    #[test]
    fn missing_reputation_scores_lower_than_positive() {
        let w = RoutingWeights::default();
        let c = card("0.0000001", "0.0000001");
        let s_unknown = weighted_score(&w, &c, None);
        let s_known = weighted_score(&w, &c, Some(rep(100, 5, 0.2)));
        assert!(
            s_known > s_unknown,
            "known reputation must outscore unknown: known={s_known}, unknown={s_unknown}"
        );
    }

    /// Equal price + equal reputation → equal score (sanity).
    #[test]
    fn identical_inputs_score_identically() {
        let w = RoutingWeights::default();
        let c = card("0.0000003", "0.0000004");
        let r = rep(500, 10, 0.3);
        let a = weighted_score(&w, &c, Some(r.clone()));
        let b = weighted_score(&w, &c, Some(r));
        assert_eq!(a, b);
    }

    /// Logs damp high-cardinality signals: doubling volume changes
    /// the score by `w.volume * ln(2)` (in the limit), not by a
    /// factor of 2. Sanity: 2× volume costs less than 1.0 of price.
    #[test]
    fn log_damping_caps_signal_growth() {
        let w = RoutingWeights { price: 1.0, volume: 1.0, renewal: 0.0, buyers: 0.0 };
        let c = card("0.0000000", "0.0000000"); // price = 0
        let s_lo = weighted_score(&w, &c, Some(rep(1, 0, 0.0)));
        let s_hi = weighted_score(&w, &c, Some(rep(1_000_000, 0, 0.0)));
        let delta = s_hi - s_lo;
        // ln(1_000_001) - ln(2) ≈ 13.13. Generous bounds.
        assert!(
            delta > 5.0 && delta < 20.0,
            "log-scale delta out of expected band: {delta}"
        );
    }
}
