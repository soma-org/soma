// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma inference {serve,proxy}` — long-running inference services.
//!
//! `serve` runs the provider-side server fronting an OpenAI-compatible
//! upstream (OpenRouter, Vast, …). `proxy` runs the local agent-facing
//! OpenAI-compatible proxy that signs requests and reconciles cost.
//!
//! Both binaries reuse the standard soma wallet (client.yaml +
//! soma.keystore) like every other `soma` subcommand — no inference-
//! specific chain plumbing.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use clap::Parser;
use sdk::wallet_context::{create_wallet_context, DEFAULT_WALLET_TIMEOUT_SEC};
use types::base::SomaAddress;
use types::config::soma_config_dir;

use inference::chain::indexer::IndexerProviderRegistry;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum InferenceCommand {
    /// Run the provider-side server.
    ///
    /// Fronts an OpenAI-compatible upstream behind SomaPay-authorized
    /// `/v1/chat/completions`. Backends today: openrouter, vast.
    #[clap(after_help = "EXAMPLE:\n    soma inference serve --config provider.toml")]
    Serve {
        /// Path to the provider TOML config (carries the model
        /// catalog + backend selection — no chain config).
        #[clap(long)]
        config: PathBuf,
        /// Address whose key signs both on-chain ops and HTTP
        /// vouchers. Defaults to the wallet's active address.
        #[clap(long)]
        address: Option<SomaAddress>,
        /// Path to the soma client config (defaults to ~/.soma/client.yaml).
        #[clap(long)]
        client: Option<PathBuf>,
        /// Soma home (for the per-channel provider ledger). Defaults
        /// to ~/.soma.
        #[clap(long)]
        soma_home: Option<PathBuf>,
        /// Heartbeat interval — `UpdateProvider` is sent on chain at
        /// this cadence so off-chain liveness watchers see a fresh
        /// `registered_at_ms`.
        #[clap(long, default_value_t = 600)]
        heartbeat_interval_secs: u64,
    },

    /// Run the local agent-facing proxy on `127.0.0.1:<port>`.
    ///
    /// Agent CLIs point at it via `OPENAI_BASE_URL`. The proxy
    /// discovers providers, picks the cheapest one for each model,
    /// and signs each request via two-tier vouchers.
    #[clap(after_help = "EXAMPLE:\n    soma inference proxy --listen 127.0.0.1:11434")]
    Proxy {
        /// Address whose key signs both on-chain ops and HTTP
        /// vouchers. Defaults to the wallet's active address.
        #[clap(long)]
        address: Option<SomaAddress>,
        /// Path to the soma client config (defaults to ~/.soma/client.yaml).
        #[clap(long)]
        client: Option<PathBuf>,
        /// Soma home (for the off-chain provider registry +
        /// per-channel client state). Defaults to ~/.soma.
        #[clap(long)]
        soma_home: Option<PathBuf>,
        /// HTTP listen address.
        #[clap(long, default_value = "127.0.0.1:11434")]
        listen: String,
        /// Default deposit (micros) when lazily opening a new channel.
        #[clap(long, default_value_t = 5_000_000)]
        default_deposit_micros: u64,
        /// Provider-list cache TTL (seconds).
        #[clap(long, default_value_t = 30)]
        provider_cache_ttl_secs: u64,
        /// Provider-routing mode: `price` (default) or `weighted`.
        /// `weighted` requires reputation from the indexer.
        #[clap(long, default_value = "price")]
        routing: String,
        /// GraphQL HTTP endpoint of the indexer. Required: provider
        /// discovery and (in `weighted` mode) reputation are sourced
        /// from here.
        #[clap(long)]
        indexer_url: String,
    },
}

impl InferenceCommand {
    pub async fn execute(self) -> Result<(), anyhow::Error> {
        match self {
            InferenceCommand::Serve {
                config,
                address,
                client,
                soma_home,
                heartbeat_interval_secs,
            } => {
                let cfg = inference::server::config::load(&config)
                    .with_context(|| format!("loading {}", config.display()))?;
                let (wallet, signer) = build_wallet(client, address).await?;
                let soma_home =
                    soma_home.map(Ok).unwrap_or_else(soma_config_dir)?;
                inference::server::run(
                    cfg,
                    wallet,
                    signer,
                    soma_home,
                    heartbeat_interval_secs,
                )
                .await
            }
            InferenceCommand::Proxy {
                address,
                client,
                soma_home,
                listen,
                default_deposit_micros,
                provider_cache_ttl_secs,
                routing,
                indexer_url,
            } => {
                let (wallet, signer) = build_wallet(client, address).await?;
                let soma_home =
                    soma_home.map(Ok).unwrap_or_else(soma_config_dir)?;
                let registry =
                    Arc::new(IndexerProviderRegistry::new(indexer_url.clone()));
                let mode = match routing.as_str() {
                    "price" => inference::proxy::config::RoutingMode::Price,
                    "weighted" => inference::proxy::config::RoutingMode::Weighted,
                    other => anyhow::bail!(
                        "unknown routing mode {other}; expected 'price' or 'weighted'"
                    ),
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
                };
                inference::proxy::run(cfg, wallet, signer, registry, soma_home).await
            }
        }
    }
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
