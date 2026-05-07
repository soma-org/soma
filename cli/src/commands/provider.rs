// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma provider {register,update,show}` — manual control over the
//! on-chain provider registry. Wraps `sdk::provider` so the CLI, the
//! inference server boot path, and any integration test all go
//! through the exact same code path.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use clap::Parser;
use sdk::wallet_context::{create_wallet_context, DEFAULT_WALLET_TIMEOUT_SEC, WalletContext};
use types::base::SomaAddress;
use types::config::soma_config_dir;
use types::provider::Provider;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum ProviderCommand {
    /// Register the active wallet address as a provider on-chain.
    /// Hard-fails if a Provider object already exists for this
    /// address — use `update` instead.
    Register {
        /// HTTP(S) endpoint at which the provider serves /soma/info
        /// and /v1/* APIs.
        #[clap(long)]
        endpoint: String,
        /// Override the active wallet address.
        #[clap(long)]
        address: Option<SomaAddress>,
        /// Override the wallet config path.
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Update the active wallet address's provider record. Bumps
    /// `registered_at_ms` (heartbeat); may also change the endpoint.
    Update {
        #[clap(long)]
        endpoint: String,
        #[clap(long)]
        address: Option<SomaAddress>,
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Print the on-chain `Provider` object as JSON. Prints `null`
    /// if the address has not registered yet.
    Show {
        /// Provider address to look up. Defaults to the active wallet
        /// address.
        #[clap(long)]
        address: Option<SomaAddress>,
        #[clap(long)]
        client: Option<PathBuf>,
    },
}

impl ProviderCommand {
    pub async fn execute(self) -> Result<()> {
        match self {
            Self::Register { endpoint, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                let id = sdk::provider::register(&mut_owned(&mut ctx), signer, endpoint).await?;
                println!("{}", id);
                Ok(())
            }
            Self::Update { endpoint, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                sdk::provider::update(&mut_owned(&mut ctx), signer, endpoint).await?;
                let id = Provider::derive_id(signer);
                println!("updated {id}");
                Ok(())
            }
            Self::Show { address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                let p = sdk::provider::get(&mut_owned(&mut ctx), signer).await?;
                let json = match p {
                    Some(p) => serde_json::to_string_pretty(&p)?,
                    None => "null".to_string(),
                };
                println!("{}", json);
                Ok(())
            }
        }
    }
}

fn build_wallet(
    client: Option<PathBuf>,
    address: Option<SomaAddress>,
) -> Result<(WalletContext, SomaAddress)> {
    let mut wallet = match client {
        Some(p) => WalletContext::new(&p)
            .with_context(|| format!("opening wallet config at {}", p.display()))?,
        None => create_wallet_context(DEFAULT_WALLET_TIMEOUT_SEC, soma_config_dir()?)?,
    };
    let signer = match address {
        Some(a) => a,
        None => wallet.active_address()?,
    };
    Ok((wallet, signer))
}

fn mut_owned(ctx: &mut WalletContext) -> &WalletContext {
    ctx
}
