// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Local OpenAI-compatible proxy. Agent CLIs (Claude Code, OpenAI SDK
//! scripts, …) point at it via `OPENAI_BASE_URL` and the proxy fans out to
//! the cheapest provider that exposes the requested model, signs each
//! request with [`crate::channel::RunningTab`], and reconciles realized
//! cost on the streamed `usage` chunk.

pub mod config;
mod relay;
mod router;
mod server;
mod state;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use fastcrypto::ed25519::Ed25519KeyPair;
use fastcrypto::traits::{KeyPair as _, ToFromBytes as _};
use sdk::keypair::Keypair;
use types::crypto::SomaKeyPair;

pub use config::Config;

use crate::chain::Discovery;
use crate::channel::RunningTab;

/// Run the proxy until shutdown. Listens on `cfg.listen.addr`.
pub async fn run(cfg: Config, kp: Keypair, chain: Arc<dyn Discovery>) -> anyhow::Result<()> {
    let cfg = Arc::new(cfg);
    let address = kp.address();
    tracing::info!(address = %address, "loaded client identity");

    let pubkey_hex = match kp.inner() {
        SomaKeyPair::Ed25519(ed) => hex::encode(ed.public().as_bytes()),
        _ => anyhow::bail!("inference proxy requires Ed25519 keys"),
    };
    let signing_kp: Ed25519KeyPair = match kp.copy_inner() {
        SomaKeyPair::Ed25519(ed) => ed,
        _ => anyhow::bail!("inference proxy requires Ed25519 keys"),
    };

    let soma_home = cfg.chain.soma_home_path();
    let store = state::ClientStore::new(soma_home.clone());
    let channel = Arc::new(RunningTab::for_client(signing_kp));

    let inner_router = Arc::new(router::Router::new(
        chain.clone(),
        store,
        cfg.clone(),
        address,
        pubkey_hex,
    ));

    // Initial discovery refresh — best-effort; proxy starts even if the
    // chain/upstreams aren't reachable yet.
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
    let listener = tokio::net::TcpListener::bind(&cfg.listen.addr)
        .await
        .with_context(|| format!("bind {}", cfg.listen.addr))?;
    tracing::info!(addr = %cfg.listen.addr, "inference proxy listening");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}
