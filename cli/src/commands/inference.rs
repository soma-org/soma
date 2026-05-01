// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma inference {serve,proxy}` — long-running inference services.
//!
//! `serve` runs the provider-side server fronting an OpenAI-compatible
//! upstream (OpenRouter, Vast, …). `proxy` runs the local agent-facing
//! OpenAI-compatible proxy that signs requests and reconciles cost.
//!
//! Both load identity from the soma keystore (active address by default,
//! or `--address`).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context as _};
use clap::Parser;
use sdk::keypair::Keypair;
use soma_keys::keystore::{AccountKeystore as _, FileBasedKeystore};
use types::base::SomaAddress;
use types::config::{soma_config_dir, SOMA_KEYSTORE_FILENAME};
use types::crypto::SomaKeyPair;

use inference::chain::local::LocalDiscovery;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum InferenceCommand {
    /// Run the provider-side server.
    ///
    /// Fronts an OpenAI-compatible upstream behind SomaPay-authorized
    /// `/v1/chat/completions`. Backends today: openrouter, vast.
    #[clap(after_help = "EXAMPLE:\n    soma inference serve --config provider.toml")]
    Serve {
        /// Path to the provider TOML config.
        #[clap(long)]
        config: PathBuf,
        /// Address whose keypair signs the channel; defaults to the keystore's active address.
        #[clap(long)]
        address: Option<SomaAddress>,
    },

    /// Run the local agent-facing proxy on `127.0.0.1:<port>`.
    ///
    /// Agent CLIs point at it via `OPENAI_BASE_URL`. The proxy discovers
    /// providers, picks the cheapest one for each model, and signs each
    /// request.
    #[clap(after_help = "EXAMPLE:\n    soma inference proxy --config client.toml")]
    Proxy {
        /// Path to the proxy TOML config.
        #[clap(long)]
        config: PathBuf,
        /// Address whose keypair signs the channel; defaults to the keystore's active address.
        #[clap(long)]
        address: Option<SomaAddress>,
    },
}

impl InferenceCommand {
    pub async fn execute(self) -> Result<(), anyhow::Error> {
        match self {
            InferenceCommand::Serve { config, address } => {
                let cfg = inference::server::config::load(&config)
                    .with_context(|| format!("loading {}", config.display()))?;
                let kp = load_keypair(address)?;
                let chain = Arc::new(
                    LocalDiscovery::new(cfg.chain.soma_home_path().join("chain"))
                        .context("init local discovery")?,
                );
                inference::server::run(cfg, kp, chain).await
            }
            InferenceCommand::Proxy { config, address } => {
                let cfg = inference::proxy::config::load(&config)
                    .with_context(|| format!("loading {}", config.display()))?;
                let kp = load_keypair(address)?;
                let chain = Arc::new(
                    LocalDiscovery::new(cfg.chain.soma_home_path().join("chain"))
                        .context("init local discovery")?,
                );
                inference::proxy::run(cfg, kp, chain).await
            }
        }
    }
}

fn load_keypair(address: Option<SomaAddress>) -> anyhow::Result<Keypair> {
    let keystore_path = soma_config_dir()?.join(SOMA_KEYSTORE_FILENAME);
    let keystore = FileBasedKeystore::load_or_create(&keystore_path)
        .with_context(|| format!("opening keystore at {}", keystore_path.display()))?;
    let addr = match address {
        Some(a) => a,
        None => *keystore
            .addresses()
            .first()
            .ok_or_else(|| anyhow!("keystore has no addresses; run `soma keytool generate` first"))?,
    };
    let kp_ref: &SomaKeyPair = keystore
        .export(&addr)
        .map_err(|e| anyhow!("export key for {addr}: {e}"))?;
    Ok(Keypair::from_inner(kp_ref.copy()))
}
