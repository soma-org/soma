// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::io::Read as _;

use anyhow::{Result, anyhow};
use clap::Parser;
use fastcrypto::hash::{Blake2b256, HashFunction as _};
use sdk::wallet_context::WalletContext;
use types::digests::ResponseDigest;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;
use crate::usdc_amount::UsdcAmount;

/// Subcommands for `soma bid`
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum BidCommand {
    /// Create a bid on an ask (offer to fulfill a task)
    Create {
        /// The ask ID to bid on
        ask_id: ObjectID,
        /// Bid price (in USDC, e.g. 1.50). Must be <= ask's max_price_per_bid.
        #[clap(long)]
        price: UsdcAmount,
        /// Response content file (hashed into response_digest). Use "-" for stdin.
        #[clap(long)]
        response: String,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
    /// List bids for an ask or by seller
    List {
        /// Filter by ask ID
        #[clap(long)]
        ask: Option<ObjectID>,
        /// Show only my bids
        #[clap(long)]
        mine: bool,
        /// Filter by status (Pending, Accepted, Rejected, Expired)
        #[clap(long)]
        status: Option<String>,
        /// Output as JSON (for agents)
        #[clap(long)]
        json: bool,
    },
    /// Poll for new bids on a specific ask (buyers listen for offers). Ctrl-C to stop.
    Listen {
        /// The ask ID to watch for bids
        ask_id: ObjectID,
        /// Poll interval (human duration: 5s, 30s, 1m). Default: 5s.
        #[clap(long, default_value = "5s")]
        interval: String,
        /// Output as JSON
        #[clap(long)]
        json: bool,
    },
}

/// Hash file content (or stdin) into a ResponseDigest using blake2b-256.
fn hash_response_content(path: &str) -> Result<ResponseDigest> {
    let content = if path == "-" {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        std::fs::read(path)
            .map_err(|e| anyhow!("Failed to read response file '{}': {}", path, e))?
    };

    let hash: [u8; 32] = Blake2b256::digest(&content).into();
    Ok(ResponseDigest::new(hash))
}

pub async fn execute(
    context: &mut WalletContext,
    cmd: BidCommand,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;

    match cmd {
        BidCommand::Create { ask_id, price, response, tx_args } => {
            let response_digest = hash_response_content(&response)?;

            let kind = TransactionKind::CreateBid(types::transaction::CreateBidArgs {
                ask_id,
                price: price.microdollars(),
                response_digest,
            });

            let (gas_ref, _) = context
                .get_richest_coin_with_balance(sender)
                .await?
                .ok_or_else(|| anyhow!("No coins found for address {}", sender))?;
            crate::client_commands::execute_or_serialize(
                context,
                sender,
                kind,
                vec![gas_ref],
                tx_args,
            )
            .await
        }
        BidCommand::List { ask, mine, status, json } => {
            let client = context.get_client().await?;

            let ask_id = ask.ok_or_else(|| {
                anyhow!("--ask is required. Usage: soma bid list --ask <ASK_ID>")
            })?;

            let bids = client
                .get_bids_for_ask(ask_id, status.as_deref())
                .await
                .map_err(|e| anyhow!("Failed to list bids: {}", e.message()))?;

            // Optionally filter to only the sender's bids
            let filtered: Vec<_> = if mine {
                bids.into_iter()
                    .filter(|obj| {
                        obj.deserialize_contents::<types::bid::Bid>(types::object::ObjectType::Bid)
                            .map(|b| b.seller == sender)
                            .unwrap_or(false)
                    })
                    .collect()
            } else {
                bids
            };

            if json {
                print_bids_json(&filtered);
            } else {
                print_bids_table(&filtered);
            }
            Ok(ClientCommandResponse::NoOutput)
        }
        BidCommand::Listen { ask_id, interval, json } => {
            let interval_ms = crate::usdc_amount::parse_duration_ms(&interval)
                .map_err(|e| anyhow!("{}", e))?;
            let interval_dur = std::time::Duration::from_millis(interval_ms);
            let client = context.get_client().await?;

            let mut seen: std::collections::HashSet<ObjectID> = std::collections::HashSet::new();

            eprintln!(
                "Listening for bids on ask {} (polling every {})... Ctrl-C to stop.",
                ask_id, interval
            );

            loop {
                let bids = client
                    .get_bids_for_ask(ask_id, None)
                    .await
                    .map_err(|e| anyhow!("Failed to poll bids: {}", e.message()))?;

                for obj in &bids {
                    let bid_id = obj.id();
                    if seen.insert(bid_id) {
                        if let Some(bid) = obj.deserialize_contents::<types::bid::Bid>(types::object::ObjectType::Bid) {
                            if json {
                                println!("{}", serde_json::to_string(&bid)?);
                            } else {
                                println!(
                                    "[NEW BID] {} — {} USDC from {} ({:?})",
                                    bid.id,
                                    bid.price as f64 / 1_000_000.0,
                                    bid.seller,
                                    bid.status,
                                );
                            }
                        }
                    }
                }

                tokio::time::sleep(interval_dur).await;
            }
        }
    }
}

fn print_bids_table(bids: &[types::object::Object]) {
    if bids.is_empty() {
        println!("No bids found.");
        return;
    }
    println!("{:<44} {:<12} {:<44} {:<10}", "BID ID", "PRICE", "SELLER", "STATUS");
    println!("{}", "-".repeat(114));
    for obj in bids {
        if let Some(bid) = obj.deserialize_contents::<types::bid::Bid>(types::object::ObjectType::Bid) {
            println!(
                "{:<44} {:<12.6} {:<44} {:<10}",
                bid.id,
                bid.price as f64 / 1_000_000.0,
                bid.seller,
                format!("{:?}", bid.status),
            );
        }
    }
}

fn print_bids_json(bids: &[types::object::Object]) {
    let items: Vec<_> = bids
        .iter()
        .filter_map(|obj| {
            obj.deserialize_contents::<types::bid::Bid>(types::object::ObjectType::Bid)
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&items).unwrap_or_default());
}
