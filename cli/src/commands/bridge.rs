// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma bridge {withdraw, status}` — user-facing CLI for the Soma ↔
//! Eth USDC bridge.
//!
//! The bridge contract surface lives in `bridge/evm/contracts/`
//! (Solidity) and `types/src/bridge.rs` (Soma-side state). Inbound
//! deposits (Eth → Soma) are driven from the EVM side (call
//! `SomaBridge.deposit(...)` on the proxy); this command only covers
//! the Soma-initiated half:
//!
//!   1. `soma bridge status` — read the live `BridgeState` (committee
//!      members + their voting power, USDC supply, pending
//!      registrations, paused flag).
//!   2. `soma bridge withdraw` — submit a `BridgeWithdraw` tx that
//!      burns USDC on the sender's accumulator and inserts a
//!      `PendingWithdrawal` shared object. The bridge nodes pick this
//!      up, collect a quorum of signatures via their HTTP sig server,
//!      attach the cert via `BridgeAttachWithdrawalSignatures`, and
//!      the outbound relayer then submits the Eth-side release tx.

use crate::usdc_amount::UsdcAmount;
use anyhow::{Result, anyhow, ensure};
use clap::{Parser, ValueEnum};
use sdk::wallet_context::WalletContext;
use std::str::FromStr;
use types::base::SomaAddress;
use types::bridge::BridgeChainId;
use types::system_state::SystemStateTrait;
use types::transaction::{BridgeWithdrawArgs, TransactionKind};

use crate::client_commands::{TxProcessingArgs, execute_or_serialize};
use crate::response::ClientCommandResponse;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum BridgeCommand {
    /// Initiate a Soma → Eth USDC withdrawal. Burns `amount` USDC from
    /// the sender's accumulator and creates a `PendingWithdrawal`
    /// shared object on chain. The bridge committee picks it up,
    /// signs, and the outbound relayer submits the Eth-side release
    /// tx within ~10–30 seconds.
    #[clap(after_help = "\
EXAMPLES:
    soma bridge withdraw --amount 1.0 \\
        --recipient 0x7B42d2B6F94fDF3c2Fe62e0aAf451487FA2DAB6e \\
        --target-chain base-sepolia")]
    Withdraw {
        /// Amount of USDC to withdraw, in decimal form (`1.5` = 1.5 USDC).
        #[clap(long)]
        amount: String,
        /// Eth recipient address (20 bytes, with or without `0x` prefix).
        #[clap(long)]
        recipient: String,
        /// Destination Eth chain. Pin the right one for the
        /// `SomaBridge` deployment you want to release from.
        #[clap(long, value_enum)]
        target_chain: BridgeTargetChain,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },

    /// Show the live `BridgeState`: pause flag, committee members and
    /// voting power, pending registrations, USDC supply, next
    /// withdrawal nonce.
    Status {
        #[clap(long, global = true, help = "Output as JSON")]
        json: bool,
    },
}

/// Subset of `BridgeChainId` the CLI exposes — the variants users
/// would actually pick as a withdrawal target. The mapping to the
/// on-chain `BridgeChainId` enum byte is wire-format-load-bearing;
/// do not renumber.
#[derive(Copy, Clone, Debug, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum BridgeTargetChain {
    EthMainnet,
    EthSepolia,
    BaseSepolia,
    EthCustom,
}

impl From<BridgeTargetChain> for BridgeChainId {
    fn from(t: BridgeTargetChain) -> Self {
        match t {
            BridgeTargetChain::EthMainnet => BridgeChainId::EthMainnet,
            BridgeTargetChain::EthSepolia => BridgeChainId::EthSepolia,
            BridgeTargetChain::BaseSepolia => BridgeChainId::BaseSepolia,
            BridgeTargetChain::EthCustom => BridgeChainId::EthCustom,
        }
    }
}

impl BridgeCommand {
    pub async fn execute(self, context: &mut WalletContext) -> Result<ClientCommandResponse> {
        match self {
            Self::Withdraw { amount, recipient, target_chain, tx_args, json: _ } => {
                let amount_micro: u64 =
                    amount.parse::<UsdcAmount>().map_err(|e| anyhow!("{e}"))?.microdollars();
                ensure!(amount_micro > 0, "Amount must be greater than 0");

                let recipient_bytes = parse_eth_address(&recipient)?;
                let sender = context.active_address()?;

                let kind = TransactionKind::BridgeWithdraw(BridgeWithdrawArgs {
                    amount: amount_micro,
                    recipient_eth_address: recipient_bytes,
                    target_chain: target_chain.into(),
                });

                execute_or_serialize(context, sender, kind, tx_args).await
            }

            Self::Status { json: _ } => {
                let client = context.get_client().await?;
                let state = client.get_latest_system_state().await?;
                let bridge_state = state.bridge_state();

                println!("BridgeState:");
                println!("  paused                  = {}", bridge_state.paused);
                println!(
                    "  total_usdc_supply       = {} micro-USDC",
                    bridge_state.total_usdc_supply
                );
                println!("  next_withdrawal_nonce   = {}", bridge_state.next_withdrawal_nonce);
                println!(
                    "  processed_deposit_count = {}",
                    bridge_state.processed_deposit_nonces.len()
                );
                println!();
                println!(
                    "Committee ({} members, thresholds: deposit={} withdraw={} pause={} unpause={}):",
                    bridge_state.bridge_committee.members.len(),
                    bridge_state.bridge_committee.threshold_deposit,
                    bridge_state.bridge_committee.threshold_withdraw,
                    bridge_state.bridge_committee.threshold_pause,
                    bridge_state.bridge_committee.threshold_unpause,
                );
                for (pk, member) in &bridge_state.bridge_committee.members {
                    println!(
                        "  pubkey=0x{} voting_power={} blocklisted={}",
                        hex::encode(pk.as_bytes()),
                        member.voting_power,
                        member.is_blocklisted,
                    );
                    println!(
                        "    soma_address=0x{} http_url={}",
                        hex::encode(member.soma_address.to_inner()),
                        member.http_url,
                    );
                }

                if !bridge_state.bridge_registrations.is_empty() {
                    println!();
                    println!(
                        "Pending registrations ({} — promoted at next epoch boundary):",
                        bridge_state.bridge_registrations.len()
                    );
                    for (addr, reg) in &bridge_state.bridge_registrations {
                        println!(
                            "  validator=0x{} pubkey=0x{} url={}",
                            hex::encode(addr.to_inner()),
                            hex::encode(reg.bridge_pubkey.as_bytes()),
                            reg.http_url,
                        );
                    }
                }

                Ok(ClientCommandResponse::NoOutput)
            }
        }
    }
}

/// Parse an Eth address from a hex string (with or without `0x`
/// prefix). Requires exactly 20 bytes — the SomaBridge contract
/// stores the recipient as a packed `address` in the
/// `TokensReleased` event payload, so the on-chain `BridgeWithdraw`
/// executor rejects anything else.
fn parse_eth_address(s: &str) -> Result<[u8; 20]> {
    let stripped = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    let bytes =
        hex::decode(stripped).map_err(|e| anyhow!("--recipient {s:?} is not valid hex: {e}"))?;
    ensure!(
        bytes.len() == 20,
        "--recipient must be exactly 20 bytes (40 hex chars); got {} bytes",
        bytes.len()
    );
    let mut arr = [0u8; 20];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_eth_address_with_prefix() {
        let addr = parse_eth_address("0x7B42d2B6F94fDF3c2Fe62e0aAf451487FA2DAB6e").unwrap();
        assert_eq!(addr[0], 0x7B);
        assert_eq!(addr[19], 0x6e);
    }

    #[test]
    fn test_parse_eth_address_no_prefix() {
        let addr = parse_eth_address("7B42d2B6F94fDF3c2Fe62e0aAf451487FA2DAB6e").unwrap();
        assert_eq!(addr[0], 0x7B);
    }

    #[test]
    fn test_parse_eth_address_wrong_length() {
        assert!(parse_eth_address("0x1234").is_err());
        assert!(parse_eth_address("0x").is_err());
    }

    #[test]
    fn test_target_chain_bytes() {
        assert_eq!(BridgeChainId::from(BridgeTargetChain::EthMainnet) as u8, 10);
        assert_eq!(BridgeChainId::from(BridgeTargetChain::EthSepolia) as u8, 11);
        assert_eq!(BridgeChainId::from(BridgeTargetChain::BaseSepolia) as u8, 13);
        assert_eq!(BridgeChainId::from(BridgeTargetChain::EthCustom) as u8, 12);
    }
}
