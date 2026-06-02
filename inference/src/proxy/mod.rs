// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Local OpenAI-compatible proxy. Agent CLIs (Claude Code, OpenAI SDK
//! scripts, …) point at it via `OPENAI_BASE_URL` and the proxy fans
//! out to the cheapest provider that exposes the requested model,
//! signs each request via [`crate::channel::RunningTab`], and
//! reconciles realized cost on the streamed `usage` chunk.

pub mod config;
mod liveness;
mod relay;
mod router;
mod server;
pub mod state;
mod trusted;
pub use liveness::LivenessCache;
pub use trusted::TrustedProviders;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ::types::base::SomaAddress;
use anyhow::Context as _;
use sdk::wallet_context::WalletContext;
use soma_keys::keystore::AccountKeystore as _;

pub use config::Config;

use crate::chain::chain::ChainChannelSurface;
use crate::chain::{ChannelSurface, ProviderRegistry};

/// Build the proxy's iroh buyer endpoint, using the **voucher-signing key**
/// as the iroh identity (n0 discovery). Because the proxy opens channels with
/// `authorized_signer = its own address`, dialing with this same key makes
/// the provider-side `peer == authorized_signer` binding pass.
async fn build_iroh_buyer(
    wallet: &WalletContext,
    address: SomaAddress,
) -> anyhow::Result<Arc<crate::transport::IrohBuyer>> {
    let keypair = wallet
        .config
        .keystore
        .export(&address)
        .with_context(|| format!("export signing key for {address}"))?;
    let secret = crate::transport::iroh_secret_from_keypair(keypair)?;
    let endpoint = crate::transport::build_n0_endpoint(secret, false).await?;
    Ok(Arc::new(crate::transport::IrohBuyer::new(endpoint)))
}

/// Run the proxy until shutdown.
pub async fn run(
    cfg: Config,
    wallet: Arc<WalletContext>,
    address: SomaAddress,
    registry: Arc<dyn ProviderRegistry>,
    _soma_home: PathBuf,
) -> anyhow::Result<()> {
    let cfg = Arc::new(cfg);
    tracing::info!(address = %address, "loaded client identity");

    let chain: Arc<dyn ChannelSurface> =
        Arc::new(ChainChannelSurface::new(wallet.clone(), address));

    // Stateless on disk — purely in-memory channel cache. On first
    // request the router rehydrates from the chain / the provider.
    let store = state::ClientStore::new();
    let channel = Arc::new(crate::channel::RunningTab::for_client(wallet.clone(), address));

    // iroh buyer endpoint (identity = voucher-signing key). Used both to
    // relay chat requests and to probe provider `/health` over iroh. Falls
    // back to a loopback endpoint if the n0 endpoint can't be brought up.
    let iroh = match build_iroh_buyer(&wallet, address).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(err = %e, "iroh buyer endpoint failed; falling back to loopback");
            Arc::new(crate::transport::IrohBuyer::bind_local().await?)
        }
    };

    let inner_router = Arc::new(router::Router::new(
        registry.clone(),
        chain.clone(),
        store,
        cfg.clone(),
        address,
        iroh.clone(),
    ));
    inner_router.spawn_liveness_refresher();

    // D. Warm the discovery cache + per-provider HTTP/2 connections
    // before serving, so the user's first chat doesn't pay for the
    // indexer query + endpoint probes. Best-effort with a short
    // timeout — if the indexer is slow or no providers are registered
    // yet, fall through; `ensure_cache` will refresh lazily on the
    // first request.
    {
        let r = inner_router.clone();
        match tokio::time::timeout(Duration::from_secs(3), r.refresh_providers()).await {
            Ok(Ok(())) => tracing::info!("discovery cache warm"),
            Ok(Err(e)) => {
                tracing::warn!(err = %e, "initial discovery refresh failed; will retry lazily")
            }
            Err(_) => tracing::warn!("initial discovery refresh timed out; will retry lazily"),
        }
    }

    // A. Pre-open a payment channel for the configured model so the
    // user's first chat doesn't pay the ~3–5 s channel-open latency.
    // Runs in the background — the listener binds immediately; if a
    // chat arrives before prewarm completes, the regular `ensure_channel`
    // path handles it (and the chain enforces single-open-per-tx-digest
    // so a race opens at most one extra channel).
    if let Some(model) = cfg.prewarm_model.clone() {
        let r = inner_router.clone();
        tokio::spawn(async move {
            // Retry until a channel for the model is ready, with a hard
            // cap so a fundamentally misconfigured proxy doesn't loop
            // forever. Two failure modes are routine and want a retry:
            //   - discovery hasn't surfaced a provider yet (provider
            //     still booting, indexer lagging)
            //   - ensure_channel failed (transient chain blip; the next
            //     attempt will pick up the channel via the indexer
            //     rehydrate path if the failed `OpenChannel` actually
            //     executed on chain)
            for attempt in 0u32..600 {
                let pick = match r.pick_provider_for_model(&model).await {
                    Ok(Some(p)) => p,
                    _ => {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                };
                let addr = pick.0.address;
                match r.ensure_channel(&pick.0, &model).await {
                    Ok(_) => {
                        tracing::info!(
                            model = %model, provider = %addr,
                            "prewarm: channel ready"
                        );
                        return;
                    }
                    Err(e) => {
                        tracing::warn!(
                            attempt, err = %e,
                            "prewarm: ensure_channel failed; will retry"
                        );
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
            tracing::warn!(model = %model, "prewarm: gave up after retries");
        });
    }

    let http = reqwest::Client::builder().timeout(Duration::from_secs(120)).build()?;

    let relay_ctx = relay::RelayCtx {
        channel: channel.clone(),
        http,
        iroh: iroh.clone(),
        wallet: wallet.clone(),
        signer: address,
    };

    let cs = Arc::new(server::ClientState { router: inner_router, relay: relay_ctx });
    let app = server::build_router(cs, cfg.local_token.clone());
    let listener = tokio::net::TcpListener::bind(&cfg.listen_addr)
        .await
        .with_context(|| format!("bind {}", cfg.listen_addr))?;
    tracing::info!(addr = %cfg.listen_addr, "inference proxy listening");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}
