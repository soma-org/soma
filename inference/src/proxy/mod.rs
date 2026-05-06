// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Local OpenAI-compatible proxy. Agent CLIs (Claude Code, OpenAI SDK
//! scripts, …) point at it via `OPENAI_BASE_URL` and the proxy fans
//! out to the cheapest provider that exposes the requested model,
//! signs each request via [`crate::channel::RunningTab`], and
//! reconciles realized cost on the streamed `usage` chunk.

pub mod config;
mod relay;
mod router;
mod server;
mod state;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use sdk::wallet_context::WalletContext;
use ::types::base::SomaAddress;

pub use config::Config;

use crate::chain::chain::ChainChannelSurface;
use crate::chain::{ChannelSurface, ProviderRegistry};

/// Run the proxy until shutdown.
pub async fn run(
    cfg: Config,
    wallet: Arc<WalletContext>,
    address: SomaAddress,
    registry: Arc<dyn ProviderRegistry>,
    soma_home: PathBuf,
) -> anyhow::Result<()> {
    let cfg = Arc::new(cfg);
    tracing::info!(address = %address, "loaded client identity");

    let chain: Arc<dyn ChannelSurface> =
        Arc::new(ChainChannelSurface::new(wallet.clone(), address));

    let store = state::ClientStore::new(soma_home);
    let channel =
        Arc::new(crate::channel::RunningTab::for_client(wallet.clone(), address));

    let inner_router = Arc::new(router::Router::new(
        registry.clone(),
        chain.clone(),
        store,
        cfg.clone(),
        address,
    ));

    // Initial discovery refresh — best-effort.
    {
        let r = inner_router.clone();
        tokio::spawn(async move {
            for _ in 0..30 {
                match r.refresh_providers().await {
                    Ok(()) => return,
                    Err(e) => tracing::debug!(err = %e, "discovery refresh failed; retry"),
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        });
    }

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let relay_ctx = relay::RelayCtx { channel: channel.clone(), http };

    let cs = Arc::new(server::ClientState { router: inner_router, relay: relay_ctx });
    let app = server::build_router(cs);
    let listener = tokio::net::TcpListener::bind(&cfg.listen_addr)
        .await
        .with_context(|| format!("bind {}", cfg.listen_addr))?;
    tracing::info!(addr = %cfg.listen_addr, "inference proxy listening");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}
