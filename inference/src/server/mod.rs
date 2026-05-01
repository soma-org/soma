// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Provider-side server: fronts an OpenAI-compatible upstream
//! ([`backend::Backend`]) behind a [`crate::channel::PaymentChannel`]-guarded
//! HTTP path.

pub mod auth;
pub mod backend;
pub mod config;
mod handler;
mod ledger;

use std::sync::Arc;

use anyhow::Context as _;
use fastcrypto::traits::{KeyPair as _, ToFromBytes as _};
use sdk::keypair::Keypair;
use types::crypto::SomaKeyPair;

pub use config::Config;

use crate::chain::types::ProviderRecord;
use crate::chain::Discovery;
use crate::channel::RunningTab;
use crate::now_ms;

/// Run the provider until shutdown.
pub async fn run(cfg: Config, kp: Keypair, chain: Arc<dyn Discovery>) -> anyhow::Result<()> {
    let address = kp.address();
    tracing::info!(address = %address, "loaded provider identity");

    let pubkey_hex = match kp.inner() {
        SomaKeyPair::Ed25519(ed) => hex::encode(ed.public().as_bytes()),
        _ => anyhow::bail!("inference provider requires Ed25519 keys"),
    };

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

    let channel = Arc::new(RunningTab::for_provider(cfg.auth.clock_skew_tolerance_secs));
    let ledger = ledger::Ledger::new(cfg.chain.soma_home_path());

    let record = ProviderRecord {
        address,
        pubkey_hex: pubkey_hex.clone(),
        endpoint: cfg.server.public_endpoint.clone(),
        last_heartbeat_ms: now_ms(),
    };
    chain.register_provider(record.clone()).await.context("register_provider on boot")?;
    spawn_heartbeat(chain.clone(), record, cfg.chain.heartbeat_interval_secs);

    let state = Arc::new(handler::ProviderState {
        chain: chain.clone(),
        backend: backend.clone(),
        channel: channel.clone(),
        ledger,
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
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

fn spawn_heartbeat(chain: Arc<dyn Discovery>, mut record: ProviderRecord, interval_secs: u64) {
    if interval_secs == 0 {
        return;
    }
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
            record.last_heartbeat_ms = now_ms();
            if let Err(e) = chain.register_provider(record.clone()).await {
                tracing::warn!(err = %e, "heartbeat register_provider failed");
            }
        }
    });
}
