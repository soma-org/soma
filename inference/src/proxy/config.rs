// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Proxy runtime config — built from CLI flags rather than a TOML
//! file. Only chain-agnostic knobs live here.

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: String,
    pub default_deposit_micros: u64,
    pub provider_cache_ttl_secs: u64,
    /// How the router picks among providers offering the same model.
    pub routing: RoutingConfig,
}

#[derive(Debug, Clone)]
pub enum RoutingMode {
    /// Cheapest provider wins. Default.
    Price,
    /// Linear combination of price (negated) and indexer-published
    /// reputation signals. See [`RoutingWeights`].
    Weighted,
}

#[derive(Debug, Clone)]
pub struct RoutingWeights {
    /// How strongly to penalize price. Larger = cheaper providers
    /// preferred. Negated internally so callers always think in
    /// "higher = better" terms.
    pub price: f64,
    /// How strongly to reward `volume_settled_30d`.
    pub volume: f64,
    /// How strongly to reward `channel_renewal_rate`.
    pub renewal: f64,
    /// How strongly to reward `distinct_buyers_30d`.
    pub buyers: f64,
}

impl Default for RoutingWeights {
    fn default() -> Self {
        Self { price: 1.0, volume: 0.3, renewal: 0.2, buyers: 0.1 }
    }
}

#[derive(Debug, Clone)]
pub struct RoutingConfig {
    pub mode: RoutingMode,
    pub weights: RoutingWeights,
    /// GraphQL HTTP endpoint of the indexer the proxy queries for
    /// reputation. `None` short-circuits weighted routing back to
    /// price-only with a one-shot warning.
    pub indexer_url: Option<String>,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            mode: RoutingMode::Price,
            weights: RoutingWeights::default(),
            indexer_url: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:11434".to_string(),
            default_deposit_micros: 5_000_000,
            provider_cache_ttl_secs: 30,
            routing: RoutingConfig::default(),
        }
    }
}
