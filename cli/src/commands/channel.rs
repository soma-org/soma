// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma channel {open,settle,top-up,request-close,withdraw,show}` —
//! manual control over on-chain payment channels. Wraps
//! `sdk::channel` so the inference proxy/server, the CLI, and any
//! integration test all go through the exact same code path.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use clap::Parser;
use sdk::wallet_context::{DEFAULT_WALLET_TIMEOUT_SEC, WalletContext, create_wallet_context};
use types::base::SomaAddress;
use types::channel::Voucher;
use types::config::soma_config_dir;
use types::crypto::GenericSignature;
use types::object::{CoinType, ObjectID};

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum ChannelCommand {
    /// Open a new channel. Returns the channel's `ObjectID`.
    Open {
        #[clap(long)]
        payee: SomaAddress,
        /// Coin type for the deposit (`usdc` or `soma`).
        #[clap(long, default_value = "usdc")]
        coin_type: CoinTypeArg,
        /// Deposit amount in base units of the coin (USDC = micros).
        #[clap(long)]
        deposit: u64,
        /// Canonical model id from the protocol ModelRegistry. The
        /// channel binds to this model for its entire lifetime; the
        /// chain executor snapshots the matching `(payee, model_id)`
        /// offering's prices + SLA bounds at open time.
        #[clap(long)]
        model_id: String,
        /// Override the active wallet address.
        #[clap(long)]
        address: Option<SomaAddress>,
        /// Override the wallet config path.
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Submit `Settle` against an existing channel. The voucher
    /// signature must already exist (typically the provider's
    /// in-memory ledger).
    Settle {
        #[clap(long)]
        channel_id: ObjectID,
        #[clap(long)]
        cumulative_amount: u64,
        /// Base64-no-pad encoded `GenericSignature` of the on-chain
        /// `Voucher{channel_id, cumulative_amount}`.
        #[clap(long)]
        signature_b64: String,
        #[clap(long)]
        address: Option<SomaAddress>,
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Top up an existing channel.
    TopUp {
        #[clap(long)]
        channel_id: ObjectID,
        #[clap(long, default_value = "usdc")]
        coin_type: CoinTypeArg,
        #[clap(long)]
        amount: u64,
        #[clap(long)]
        address: Option<SomaAddress>,
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Begin the close timer for an existing channel.
    RequestClose {
        #[clap(long)]
        channel_id: ObjectID,
        #[clap(long)]
        address: Option<SomaAddress>,
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Withdraw remainder after the grace period elapses.
    Withdraw {
        #[clap(long)]
        channel_id: ObjectID,
        #[clap(long)]
        address: Option<SomaAddress>,
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// Print the on-chain `Channel` object as JSON.
    Show {
        #[clap(long)]
        channel_id: ObjectID,
        #[clap(long)]
        client: Option<PathBuf>,
    },
    /// List the active wallet's channels by querying the indexer.
    /// Recovers buyer-side channel state without any local files.
    List {
        /// Filter by role: `payer` or `payee`. Default: payer.
        #[clap(long, default_value = "payer")]
        role: String,
        /// Filter by status: `open`, `closing`, or `withdrawn`.
        /// Omit for all.
        #[clap(long)]
        status: Option<String>,
        /// GraphQL HTTP endpoint of the indexer.
        #[clap(long)]
        indexer_url: String,
        /// Override the wallet address being queried for.
        #[clap(long)]
        address: Option<SomaAddress>,
        /// Override the wallet config path.
        #[clap(long)]
        client: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum CoinTypeArg {
    Usdc,
    Soma,
}

impl std::str::FromStr for CoinTypeArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "usdc" => Ok(Self::Usdc),
            "soma" => Ok(Self::Soma),
            other => Err(format!("unknown coin_type {other}; expected 'usdc' or 'soma'")),
        }
    }
}

impl From<CoinTypeArg> for CoinType {
    fn from(c: CoinTypeArg) -> Self {
        match c {
            CoinTypeArg::Usdc => CoinType::Usdc,
            CoinTypeArg::Soma => CoinType::Soma,
        }
    }
}

impl ChannelCommand {
    pub async fn execute(self) -> Result<()> {
        match self {
            Self::Open { payee, coin_type, deposit, model_id, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                let coin_type: CoinType = coin_type.into();
                let id = sdk::channel::open_channel(
                    &mut_owned(&mut ctx),
                    signer,
                    payee,
                    signer,
                    coin_type,
                    deposit,
                    model_id,
                )
                .await?;
                println!("{}", id);
                Ok(())
            }
            Self::Settle { channel_id, cumulative_amount, signature_b64, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                // CLI `settle` only carries the cumulative amount; the
                // usage breakdown defaults to zero. Producer-side
                // tooling that signs vouchers with real usage uses the
                // SDK directly.
                let voucher = Voucher::new_amount_only(channel_id, cumulative_amount);
                let sig = decode_sig(&signature_b64)?;
                sdk::channel::settle(&mut_owned(&mut ctx), signer, voucher, sig).await?;
                println!("settled {channel_id} at {cumulative_amount}");
                Ok(())
            }
            Self::TopUp { channel_id, coin_type, amount, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                sdk::channel::top_up(
                    &mut_owned(&mut ctx),
                    signer,
                    channel_id,
                    coin_type.into(),
                    amount,
                )
                .await?;
                println!("topped up {channel_id} by {amount}");
                Ok(())
            }
            Self::RequestClose { channel_id, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                sdk::channel::request_close(&mut_owned(&mut ctx), signer, channel_id).await?;
                println!("request_close submitted for {channel_id}");
                Ok(())
            }
            Self::Withdraw { channel_id, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                sdk::channel::withdraw_after_timeout(&mut_owned(&mut ctx), signer, channel_id)
                    .await?;
                println!("withdrew {channel_id}");
                Ok(())
            }
            Self::Show { channel_id, client } => {
                let (ctx, _) = build_wallet(client, None)?;
                let client = ctx.get_client().await?;
                let obj = client
                    .get_object(channel_id)
                    .await
                    .map_err(|e| anyhow::anyhow!("get_object: {e}"))?;
                let chan = obj
                    .as_channel()
                    .ok_or_else(|| anyhow::anyhow!("{channel_id} is not a Channel"))?;
                println!("{}", serde_json::to_string_pretty(&chan)?);
                Ok(())
            }
            Self::List { role, status, indexer_url, address, client } => {
                let (_ctx, signer) = build_wallet(client, address)?;
                let role_arg = match role.as_str() {
                    "payer" => "payer",
                    "payee" => "payee",
                    other => anyhow::bail!("unknown role {other}; expected 'payer' or 'payee'"),
                };
                let status_filter = match status.as_deref() {
                    None => "",
                    Some("open") => ", status: OPEN",
                    Some("closing") => ", status: CLOSING",
                    Some("withdrawn") => ", status: WITHDRAWN",
                    Some(other) => anyhow::bail!(
                        "unknown status {other}; expected 'open' / 'closing' / 'withdrawn'"
                    ),
                };
                let addr_hex = format!("0x{}", hex::encode(signer.to_vec()));
                let query = format!(
                    r#"query Channels($a: String!) {{
                        channels({role_arg}: $a{status_filter}, first: 50) {{
                            edges {{ node {{
                                id payer payee token deposit settledAmount
                                status closeRequestedAtMs lastUpdateCp
                            }} }}
                        }}
                    }}"#
                );
                let body = serde_json::json!({
                    "query": query,
                    "variables": { "a": addr_hex },
                });
                let resp = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(10))
                    .build()?
                    .post(&indexer_url)
                    .json(&body)
                    .send()
                    .await?;
                if !resp.status().is_success() {
                    anyhow::bail!("indexer returned status {}", resp.status());
                }
                let v: serde_json::Value = resp.json().await?;
                println!("{}", serde_json::to_string_pretty(&v)?);
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

/// `WalletContext::active_address` takes `&mut self` so we wrap to
/// hand a `&WalletContext` (immutable) to the SDK helpers.
fn mut_owned(ctx: &mut WalletContext) -> &WalletContext {
    ctx
}

fn decode_sig(b64: &str) -> Result<GenericSignature> {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use fastcrypto::traits::ToFromBytes as _;
    let bytes = URL_SAFE_NO_PAD.decode(b64.as_bytes()).with_context(|| "decode signature_b64")?;
    GenericSignature::from_bytes(&bytes).map_err(|e| anyhow::anyhow!("parse GenericSignature: {e}"))
}
