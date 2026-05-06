// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Provider-side server: fronts an OpenAI-compatible upstream
//! ([`backend::Backend`]) behind a [`crate::channel::PaymentChannel`]-guarded
//! HTTP path.

pub mod auth;
pub mod backend;
pub mod config;
mod handler;
pub mod ledger;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use sdk::wallet_context::WalletContext;
use ::types::base::SomaAddress;

pub use config::Config;

use crate::chain::chain::ChainChannelSurface;
use crate::chain::{ChannelSurface, ProviderRegistry, ProviderRecord};
use crate::channel::RunningTab;
use crate::now_ms;

/// Run the provider until shutdown (SIGTERM / SIGINT).
///
/// On graceful shutdown, every channel with a stored on-chain
/// signature is settled — the provider claims its earned share
/// before exiting.
pub async fn run(
    cfg: Config,
    wallet: Arc<WalletContext>,
    address: SomaAddress,
    registry: Arc<dyn ProviderRegistry>,
    soma_home: PathBuf,
    heartbeat_interval_secs: u64,
) -> anyhow::Result<()> {
    tracing::info!(address = %address, "loaded provider identity");

    // Pubkey hex is provider metadata only; clients don't need it for
    // signature verification any more (the on-chain Channel carries
    // the authorized_signer address and signatures verify against it).
    // Kept for the /soma/info endpoint.
    let pubkey_hex = String::new();

    let backend: Arc<dyn backend::Backend> = match cfg.backend.kind.as_str() {
        "openrouter" => backend::openrouter::OpenRouterBackend::new(&cfg)?,
        "vast" => backend::vast::VastBackend::new(&cfg)?,
        other => anyhow::bail!("unsupported backend kind: {other}"),
    };

    let mut catalog: Vec<crate::catalog::ModelCard> = backend
        .list_models()
        .await
        .context("backend list_models on boot")?
        .data;
    backend::fill_soma_info(&mut catalog, &address.to_string());
    if !cfg.offerings.is_empty() {
        catalog = cfg.offerings.clone();
    }

    let chain: Arc<dyn ChannelSurface> =
        Arc::new(ChainChannelSurface::new(wallet.clone(), address));

    let channel = Arc::new(RunningTab::for_provider(cfg.auth.clock_skew_tolerance_secs));
    let ledger = ledger::Ledger::new(soma_home);

    let record = ProviderRecord {
        address,
        pubkey_hex: pubkey_hex.clone(),
        endpoint: cfg.server.public_endpoint.clone(),
        last_heartbeat_ms: now_ms(),
    };
    registry.register_provider(record.clone()).await.context("register_provider on boot")?;
    spawn_heartbeat(registry.clone(), record, heartbeat_interval_secs);

    let state = Arc::new(handler::ProviderState {
        chain: chain.clone(),
        backend: backend.clone(),
        channel: channel.clone(),
        ledger: ledger.clone(),
        catalog,
        provider_address: address.to_string(),
        provider_pubkey_hex: pubkey_hex,
        public_endpoint: cfg.server.public_endpoint.clone(),
    });

    let app = handler::build_router(state);
    let listener = tokio::net::TcpListener::bind(&cfg.server.listen)
        .await
        .with_context(|| format!("bind {}", cfg.server.listen))?;
    tracing::info!(addr = %cfg.server.listen, "inference server listening");

    let serve = axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal(chain.clone(), ledger.clone(), channel.clone()));
    serve.await?;
    Ok(())
}

fn spawn_heartbeat(
    registry: Arc<dyn ProviderRegistry>,
    mut record: ProviderRecord,
    interval_secs: u64,
) {
    if interval_secs == 0 {
        return;
    }
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
            record.last_heartbeat_ms = now_ms();
            if let Err(e) = registry.register_provider(record.clone()).await {
                tracing::warn!(err = %e, "heartbeat register_provider failed");
            }
        }
    });
}

/// Wait for SIGTERM/SIGINT, then walk the ledger and settle every
/// channel that has a stored on-chain signature. We use the SDK's
/// `settle` helper so the on-chain side is identical to a manual
/// `soma channel settle` call.
async fn shutdown_signal(
    chain: Arc<dyn ChannelSurface>,
    ledger: ledger::Ledger,
    channel: Arc<RunningTab>,
) {
    use crate::channel::PaymentChannel as _;
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
    tracing::info!("shutdown signal received; settling open channels…");

    let snapshot = ledger.snapshot().await;
    for (id, state) in snapshot {
        let pair = channel.final_settlement(&state);
        let Some((voucher, sig)) = pair else {
            tracing::info!(channel = %id, "no signature held; skipping settle");
            continue;
        };
        match chain.settle(voucher, sig).await {
            Ok(()) => tracing::info!(channel = %id, cumulative = state.cumulative_authorized_micros, "settled on shutdown"),
            Err(e) => tracing::warn!(channel = %id, err = %e, "settle on shutdown failed"),
        }
    }
}
