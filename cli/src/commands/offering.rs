// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma offering {register,update,deactivate,show}` — manual control
//! over a provider's on-chain per-(provider, model) `Offering` shared
//! object. Wraps `sdk::offering` so the CLI, the inference server, and
//! integration tests all share the same code path.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use clap::Parser;
use sdk::offering::OfferingPrices;
use sdk::wallet_context::{create_wallet_context, DEFAULT_WALLET_TIMEOUT_SEC, WalletContext};
use types::base::SomaAddress;
use types::config::soma_config_dir;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum OfferingCommand {
    /// Register a new offering for `--model-id` against the active
    /// wallet address. The chain executor validates `model_id` against
    /// the protocol-config `ModelRegistry`; unknown ids are rejected.
    Register {
        /// Canonical model id (must be in the ModelRegistry at the
        /// active protocol version). Run `soma model list` to see the
        /// allowed values.
        #[clap(long)]
        model_id: String,
        /// Price for input/prompt tokens, in micros per 1000 tokens.
        /// (1 micro = $1e-6.)
        #[clap(long)]
        prompt_micros_per_1k: u64,
        /// Price for output/completion tokens, in micros per 1000 tokens.
        #[clap(long)]
        completion_micros_per_1k: u64,
        /// Price for cache-read prompt tokens, in micros per 1000 tokens.
        /// Defaults to 0 (no caching surcharge).
        #[clap(long, default_value_t = 0)]
        cache_read_micros_per_1k: u64,
        /// Price for cache-write prompt tokens, in micros per 1000 tokens.
        /// Defaults to 0.
        #[clap(long, default_value_t = 0)]
        cache_write_micros_per_1k: u64,
        /// Flat per-request surcharge (micros). Defaults to 0.
        #[clap(long, default_value_t = 0)]
        request_micros: u64,
        /// Time-to-first-token SLA bound (ms). Channels opened against
        /// this offering snapshot this value; the relay emits a
        /// negative `RateChannel(TtftBreach)` if a response exceeds it.
        /// `0` disables the check.
        #[clap(long, default_value_t = 0)]
        ttft_bound_ms: u32,
        /// Time-to-output-token SLA bound (ms per output token).
        /// `0` disables.
        #[clap(long, default_value_t = 0)]
        ttot_bound_ms: u32,
        #[clap(long)]
        address: Option<SomaAddress>,
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Update the offering for `(active wallet, --model-id)`. Same
    /// fields as `register`; re-activates a previously-deactivated row.
    Update {
        #[clap(long)]
        model_id: String,
        #[clap(long)]
        prompt_micros_per_1k: u64,
        #[clap(long)]
        completion_micros_per_1k: u64,
        #[clap(long, default_value_t = 0)]
        cache_read_micros_per_1k: u64,
        #[clap(long, default_value_t = 0)]
        cache_write_micros_per_1k: u64,
        #[clap(long, default_value_t = 0)]
        request_micros: u64,
        #[clap(long, default_value_t = 0)]
        ttft_bound_ms: u32,
        #[clap(long, default_value_t = 0)]
        ttot_bound_ms: u32,
        #[clap(long)]
        address: Option<SomaAddress>,
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Deactivate the offering for `(active wallet, --model-id)`. The
    /// row stays on chain for audit; new channels against this pair
    /// are rejected until a follow-up `update` re-activates it.
    Deactivate {
        #[clap(long)]
        model_id: String,
        #[clap(long)]
        address: Option<SomaAddress>,
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Print the on-chain `Offering` for `(--provider, --model-id)` as
    /// JSON. Prints `null` if no Offering exists for the pair.
    Show {
        /// Provider address. Defaults to the active wallet address.
        #[clap(long)]
        provider: Option<SomaAddress>,
        #[clap(long)]
        model_id: String,
        #[clap(long)]
        client: Option<PathBuf>,
    },
}

impl OfferingCommand {
    pub async fn execute(self) -> Result<()> {
        match self {
            Self::Register {
                model_id,
                prompt_micros_per_1k,
                completion_micros_per_1k,
                cache_read_micros_per_1k,
                cache_write_micros_per_1k,
                request_micros,
                ttft_bound_ms,
                ttot_bound_ms,
                address,
                client,
            } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                let id = sdk::offering::register(
                    &mut_owned(&mut ctx),
                    signer,
                    model_id,
                    OfferingPrices {
                        prompt_micros_per_1k,
                        completion_micros_per_1k,
                        cache_read_micros_per_1k,
                        cache_write_micros_per_1k,
                        request_micros,
                        ttft_bound_ms,
                        ttot_bound_ms,
                    },
                )
                .await?;
                println!("{}", id);
                Ok(())
            }
            Self::Update {
                model_id,
                prompt_micros_per_1k,
                completion_micros_per_1k,
                cache_read_micros_per_1k,
                cache_write_micros_per_1k,
                request_micros,
                ttft_bound_ms,
                ttot_bound_ms,
                address,
                client,
            } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                sdk::offering::update(
                    &mut_owned(&mut ctx),
                    signer,
                    model_id.clone(),
                    OfferingPrices {
                        prompt_micros_per_1k,
                        completion_micros_per_1k,
                        cache_read_micros_per_1k,
                        cache_write_micros_per_1k,
                        request_micros,
                        ttft_bound_ms,
                        ttot_bound_ms,
                    },
                )
                .await?;
                let id = types::offering::Offering::derive_id(signer, &model_id);
                println!("updated {id}");
                Ok(())
            }
            Self::Deactivate { model_id, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                sdk::offering::deactivate(&mut_owned(&mut ctx), signer, model_id.clone()).await?;
                let id = types::offering::Offering::derive_id(signer, &model_id);
                println!("deactivated {id}");
                Ok(())
            }
            Self::Show { provider, model_id, client } => {
                let (mut ctx, signer) = build_wallet(client, provider)?;
                let o = sdk::offering::get(&mut_owned(&mut ctx), signer, &model_id).await?;
                let json = match o {
                    Some(o) => serde_json::to_string_pretty(&o)?,
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
