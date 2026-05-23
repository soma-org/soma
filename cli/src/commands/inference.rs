// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Long-running inference daemons surfaced as `soma start {provider, proxy}`.
//!
//! - `start provider` runs the provider-side server fronting an OpenAI-compatible
//!   upstream (OpenRouter, Vast, …).
//! - `start proxy` runs the local agent-facing OpenAI-compatible proxy that signs
//!   requests and reconciles cost.
//!
//! Both reuse the standard soma wallet (`client.yaml` + `soma.keystore`) like every
//! other `soma` subcommand — no inference-specific chain plumbing.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use clap::Args;
use sdk::wallet_context::{DEFAULT_WALLET_TIMEOUT_SEC, create_wallet_context};
use types::base::SomaAddress;
use types::config::soma_config_dir;

use inference::chain::indexer::IndexerProviderRegistry;

#[derive(Args, Debug)]
pub struct ProviderArgs {
    /// Path to the provider TOML config (carries the model catalog +
    /// backend selection — no chain config).
    #[clap(long)]
    pub config: PathBuf,
    /// Address whose key signs both on-chain ops and HTTP vouchers.
    /// Defaults to the wallet's active address.
    #[clap(long)]
    pub address: Option<SomaAddress>,
    /// Path to the soma client config (defaults to ~/.soma/client.yaml).
    #[clap(long)]
    pub client: Option<PathBuf>,
    /// Soma home (for the per-channel provider ledger). Defaults to ~/.soma.
    #[clap(long)]
    pub soma_home: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct ProxyArgs {
    /// Address whose key signs both on-chain ops and HTTP vouchers.
    /// Defaults to the wallet's active address.
    #[clap(long)]
    pub address: Option<SomaAddress>,
    /// Path to the soma client config (defaults to ~/.soma/client.yaml).
    #[clap(long)]
    pub client: Option<PathBuf>,
    /// Soma home (for the off-chain provider registry + per-channel
    /// client state). Defaults to ~/.soma.
    #[clap(long)]
    pub soma_home: Option<PathBuf>,
    /// HTTP listen address.
    #[clap(long, default_value = "127.0.0.1:11434")]
    pub listen: String,
    /// Default deposit (micros) when lazily opening a new channel.
    #[clap(long, default_value_t = 5_000_000)]
    pub default_deposit_micros: u64,
    /// Provider-list cache TTL (seconds).
    #[clap(long, default_value_t = 30)]
    pub provider_cache_ttl_secs: u64,
    /// Provider-routing mode: `price` (default) or `weighted`. `weighted`
    /// requires reputation from the indexer.
    #[clap(long, default_value = "price")]
    pub routing: String,
    /// GraphQL HTTP endpoint of the indexer. Required: provider
    /// discovery and (in `weighted` mode) reputation are sourced from
    /// here.
    #[clap(long)]
    pub indexer_url: String,
    /// HTTP endpoint of the points service that publishes the per-season
    /// authorized-providers allowlist (e.g. `https://points.soma.org`).
    /// When set, the proxy refreshes the allowlist periodically and
    /// skips routing to any provider not on it. somacode defaults to ON
    /// via the `points.soma.org` endpoint; set `--include-untrusted` to
    /// opt out.
    #[clap(long)]
    pub trusted_providers_url: Option<String>,
    /// Refresh interval (seconds) for the trusted-providers snapshot.
    /// Defaults to 10 minutes — short enough to pick up a fresh
    /// authorization quickly without hammering the points service.
    #[clap(long, default_value_t = 600)]
    pub trusted_providers_refresh_secs: u64,
    /// Disable the trusted-providers filter entirely. Routes through
    /// every provider that satisfies the price / SLA filters regardless
    /// of authorization status. Use this in localnet, in tests, and
    /// when running with a custom `--indexer-url` whose providers aren't
    /// on the public points list.
    #[clap(long)]
    pub include_untrusted: bool,
    /// On boot, after discovery surfaces a provider for this model,
    /// pre-open a payment channel against it — so the first user
    /// request doesn't pay the ~3–5 s channel-open latency. Omit to
    /// keep the lazy-on-first-request behaviour.
    #[clap(long)]
    pub prewarm_model: Option<String>,
    /// Disable the per-provider liveness probe. Default is to probe
    /// each provider's `/health` endpoint with a short timeout and
    /// skip ones that don't respond before routing. Turn off when
    /// running against a fixture / localnet provider that doesn't
    /// expose `/health`.
    #[clap(long)]
    pub no_probe_liveness: bool,
    /// Refresh cadence (seconds) for the liveness cache. Probes
    /// older than this are re-issued on demand. Lower-bounded to 5 s
    /// internally.
    #[clap(long, default_value_t = 30)]
    pub liveness_refresh_secs: u64,
    /// Per-probe HTTP timeout (milliseconds). Lower-bounded to
    /// 100 ms internally. The default is short enough that a single
    /// bad provider doesn't add noticeable latency to a chat
    /// request that has to fall through to the next candidate.
    #[clap(long, default_value_t = 1_500)]
    pub liveness_timeout_ms: u64,
}

pub async fn run_provider(args: ProviderArgs) -> Result<(), anyhow::Error> {
    let ProviderArgs { config, address, client, soma_home } = args;
    let cfg = inference::server::config::load(&config)
        .with_context(|| format!("loading {}", config.display()))?;
    let (wallet, signer) = build_wallet(client, address).await?;
    let soma_home = soma_home.map(Ok).unwrap_or_else(soma_config_dir)?;
    inference::server::run(cfg, wallet, signer, soma_home).await
}

pub async fn run_proxy(args: ProxyArgs) -> Result<(), anyhow::Error> {
    let ProxyArgs {
        address,
        client,
        soma_home,
        listen,
        default_deposit_micros,
        provider_cache_ttl_secs,
        routing,
        indexer_url,
        trusted_providers_url,
        trusted_providers_refresh_secs,
        include_untrusted,
        prewarm_model,
        no_probe_liveness,
        liveness_refresh_secs,
        liveness_timeout_ms,
    } = args;
    let (wallet, signer) = build_wallet(client, address).await?;
    let soma_home = soma_home.map(Ok).unwrap_or_else(soma_config_dir)?;
    let registry = Arc::new(IndexerProviderRegistry::new(indexer_url.clone()));
    let mode = match routing.as_str() {
        "price" => inference::proxy::config::RoutingMode::Price,
        "weighted" => inference::proxy::config::RoutingMode::Weighted,
        other => anyhow::bail!("unknown routing mode {other}; expected 'price' or 'weighted'"),
    };
    let cfg = inference::proxy::Config {
        listen_addr: listen,
        default_deposit_micros,
        provider_cache_ttl_secs,
        routing: inference::proxy::config::RoutingConfig {
            mode,
            weights: Default::default(),
            indexer_url: Some(indexer_url),
        },
        // Trusted-providers filter on by default (somacode posture);
        // `--include-untrusted` opts out.
        trusted_providers_only: !include_untrusted,
        trusted_providers_url,
        trusted_providers_refresh_secs,
        prewarm_model,
        probe_liveness: !no_probe_liveness,
        liveness_refresh_secs,
        liveness_timeout_ms,
    };
    inference::proxy::run(cfg, wallet, signer, registry, soma_home).await
}

async fn build_wallet(
    client: Option<PathBuf>,
    address: Option<SomaAddress>,
) -> anyhow::Result<(Arc<sdk::wallet_context::WalletContext>, SomaAddress)> {
    let wallet = match client {
        Some(p) => sdk::wallet_context::WalletContext::new(&p)
            .with_context(|| format!("opening wallet config at {}", p.display()))?,
        None => create_wallet_context(DEFAULT_WALLET_TIMEOUT_SEC, soma_config_dir()?)?,
    };
    let mut wallet = wallet;
    let signer = match address {
        Some(a) => a,
        None => wallet.active_address()?,
    };
    Ok((Arc::new(wallet), signer))
}
