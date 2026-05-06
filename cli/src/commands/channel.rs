// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma channel {open,settle,top-up,request-close,withdraw,show}` —
//! manual control over on-chain payment channels. Wraps
//! `sdk::channel` so the inference proxy/server, the CLI, and any
//! integration test all go through the exact same code path.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use clap::Parser;
use sdk::wallet_context::{create_wallet_context, DEFAULT_WALLET_TIMEOUT_SEC, WalletContext};
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
            Self::Open { payee, coin_type, deposit, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                let coin_type: CoinType = coin_type.into();
                let id = sdk::channel::open_channel(
                    &mut_owned(&mut ctx),
                    signer,
                    payee,
                    signer,
                    coin_type,
                    deposit,
                )
                .await?;
                println!("{}", id);
                Ok(())
            }
            Self::Settle { channel_id, cumulative_amount, signature_b64, address, client } => {
                let (mut ctx, signer) = build_wallet(client, address)?;
                let voucher = Voucher::new(channel_id, cumulative_amount);
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
                let obj = client.get_object(channel_id).await
                    .map_err(|e| anyhow::anyhow!("get_object: {e}"))?;
                let chan = obj.as_channel()
                    .ok_or_else(|| anyhow::anyhow!("{channel_id} is not a Channel"))?;
                println!("{}", serde_json::to_string_pretty(&chan)?);
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
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use fastcrypto::traits::ToFromBytes as _;
    let bytes = URL_SAFE_NO_PAD
        .decode(b64.as_bytes())
        .with_context(|| "decode signature_b64")?;
    GenericSignature::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("parse GenericSignature: {e}"))
}
