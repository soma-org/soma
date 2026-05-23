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
    /// When `true`, the router restricts its candidate set to the
    /// provider addresses returned by
    /// `GET <trusted_providers_url>/v1/providers/authorized`.
    /// somacode flips this on by default; advanced users opt out via
    /// `--include-untrusted`. When `trusted_providers_url` is `None`
    /// the flag is a no-op (filter is open).
    pub trusted_providers_only: bool,
    pub trusted_providers_url: Option<String>,
    /// Refresh cadence for the trusted-providers fetch.
    pub trusted_providers_refresh_secs: u64,
    /// On boot, after discovery surfaces a provider for this model,
    /// pre-open a payment channel against it — so the user's first
    /// chat doesn't pay the ~3–5 s channel-open latency. `None` keeps
    /// the lazy-on-first-request behaviour.
    pub prewarm_model: Option<String>,
    /// When `true`, before scoring candidates the router probes each
    /// provider's `/health` endpoint and skips ones that don't
    /// respond within `liveness_timeout_ms`. Results are cached for
    /// `liveness_refresh_secs`. On by default — without it, a stale
    /// on-chain registration pointing at a dead endpoint will be
    /// picked for routing and surface as an opaque `channel_error`
    /// to the caller.
    pub probe_liveness: bool,
    pub liveness_refresh_secs: u64,
    pub liveness_timeout_ms: u64,
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
    /// How strongly to *penalize* `negative_rate_30d` (the
    /// volume-weighted fraction of recent ratings that were
    /// negative). Subtracted from the score, so larger = more
    /// aggressive avoidance of complained-about providers.
    /// `None` ratings (no in-window data) contribute zero
    /// regardless of the weight.
    pub negative_rate: f64,
}

impl Default for RoutingWeights {
    fn default() -> Self {
        Self { price: 1.0, volume: 0.3, renewal: 0.2, buyers: 0.1, negative_rate: 1.0 }
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
        Self { mode: RoutingMode::Price, weights: RoutingWeights::default(), indexer_url: None }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:11434".to_string(),
            default_deposit_micros: 5_000_000,
            provider_cache_ttl_secs: 30,
            routing: RoutingConfig::default(),
            // Default-on: somacode-style routing is the production
            // posture. Power users override via --include-untrusted.
            // When no `trusted_providers_url` is configured, this
            // flag is a no-op (filter is open).
            trusted_providers_only: true,
            trusted_providers_url: None,
            trusted_providers_refresh_secs: 600,
            prewarm_model: None,
            probe_liveness: true,
            liveness_refresh_secs: 30,
            liveness_timeout_ms: 1_500,
        }
    }
}
