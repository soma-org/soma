// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::io::Read as _;

use anyhow::{Result, anyhow};
use clap::Parser;
use fastcrypto::hash::{Blake2b256, HashFunction as _};
use sdk::wallet_context::WalletContext;
use types::digests::TaskDigest;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;
use crate::usdc_amount::{UsdcAmount, parse_duration_ms};

/// Subcommands for `soma ask`
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum AskCommand {
    /// Post a new ask (task request)
    Create {
        /// Task content file (hashed into task_digest). Use "-" for stdin.
        #[clap(long)]
        task: String,
        /// Maximum price per bid (in USDC, e.g. 1.50)
        #[clap(long)]
        max_price: UsdcAmount,
        /// Number of bids the buyer intends to accept (default: 1)
        #[clap(long, default_value = "1")]
        num_bids: u32,
        /// Timeout for bids (human duration: 30s, 5m, 1h, 1d). Default: 5m.
        #[clap(long, default_value = "5m")]
        timeout: String,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
    /// Cancel an open ask before any bids are accepted
    Cancel {
        /// The ask ID to cancel
        ask_id: ObjectID,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
    /// List open asks (for sellers to discover work)
    List {
        /// Filter by buyer address
        #[clap(long)]
        buyer: Option<String>,
        /// Maximum results to return
        #[clap(long, default_value = "100")]
        limit: u32,
        /// Output as JSON (for agents)
        #[clap(long)]
        json: bool,
    },
    /// Get details of a specific ask
    Info {
        /// The ask ID
        ask_id: ObjectID,
        /// Output as JSON
        #[clap(long)]
        json: bool,
    },
    /// Poll for new open asks (sellers listen for work). Ctrl-C to stop.
    Listen {
        /// Poll interval (human duration: 5s, 30s, 1m). Default: 5s.
        #[clap(long, default_value = "5s")]
        interval: String,
        /// Output as JSON
        #[clap(long)]
        json: bool,
    },
}

/// Hash file content (or stdin) into a TaskDigest using blake2b-256.
pub fn hash_task_content(path: &str) -> Result<TaskDigest> {
    let content = if path == "-" {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        std::fs::read(path).map_err(|e| anyhow!("Failed to read task file '{}': {}", path, e))?
    };

    let hash: [u8; 32] = Blake2b256::digest(&content).into();
    Ok(TaskDigest::new(hash))
}

pub async fn execute(
    context: &mut WalletContext,
    cmd: AskCommand,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;

    match cmd {
        AskCommand::Create { task, max_price, num_bids, timeout, tx_args } => {
            let task_digest = hash_task_content(&task)?;
            let timeout_ms =
                parse_duration_ms(&timeout).map_err(|e| anyhow!("{}", e))?;

            let kind = TransactionKind::CreateAsk(types::transaction::CreateAskArgs {
                task_digest,
                max_price_per_bid: max_price.microdollars(),
                num_bids_wanted: num_bids,
                timeout_ms,
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
        AskCommand::Cancel { ask_id, tx_args } => {
            let kind = TransactionKind::CancelAsk { ask_id };
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
        AskCommand::List { buyer, limit, json } => {
            let client = context.get_client().await?;
            let buyer_addr = buyer
                .as_ref()
                .map(|b| b.parse::<types::base::SomaAddress>())
                .transpose()
                .map_err(|_| anyhow!("Invalid buyer address"))?;

            let asks = client
                .get_open_asks(buyer_addr.as_ref(), Some(limit))
                .await
                .map_err(|e| anyhow!("Failed to list asks: {}", e.message()))?;

            if json {
                print_asks_json(&asks);
            } else {
                print_asks_table(&asks);
            }
            Ok(ClientCommandResponse::NoOutput)
        }
        AskCommand::Info { ask_id, json } => {
            let client = context.get_client().await?;
            let obj = client
                .get_ask(ask_id)
                .await
                .map_err(|e| anyhow!("Failed to get ask: {}", e.message()))?;

            if let Some(ask) = obj.deserialize_contents::<types::ask::Ask>(types::object::ObjectType::Ask) {
                if json {
                    println!("{}", serde_json::to_string_pretty(&ask)?);
                } else {
                    println!("Ask {}", ask.id);
                    println!("  Buyer:           {}", ask.buyer);
                    println!("  Task Digest:     {}", ask.task_digest);
                    println!("  Max Price/Bid:   {} USDC", ask.max_price_per_bid as f64 / 1_000_000.0);
                    println!("  Num Bids Wanted: {}", ask.num_bids_wanted);
                    println!("  Accepted:        {}", ask.accepted_bid_count);
                    println!("  Timeout:         {}ms", ask.timeout_ms);
                    println!("  Created At:      {}ms", ask.created_at_ms);
                    println!("  Status:          {:?}", ask.status);
                }
            } else {
                return Err(anyhow!("Failed to deserialize ask object {}", ask_id));
            }
            Ok(ClientCommandResponse::NoOutput)
        }
        AskCommand::Listen { interval, json } => {
            let interval_ms = parse_duration_ms(&interval).map_err(|e| anyhow!("{}", e))?;
            let interval_dur = std::time::Duration::from_millis(interval_ms);
            let client = context.get_client().await?;

            let mut seen: std::collections::HashSet<ObjectID> = std::collections::HashSet::new();

            eprintln!("Listening for new asks (polling every {})... Ctrl-C to stop.", interval);

            loop {
                let asks = client
                    .get_open_asks(None, Some(200))
                    .await
                    .map_err(|e| anyhow!("Failed to poll asks: {}", e.message()))?;

                for obj in &asks {
                    let ask_id = obj.id();
                    if seen.insert(ask_id) {
                        if let Some(ask) = obj.deserialize_contents::<types::ask::Ask>(types::object::ObjectType::Ask) {
                            if json {
                                println!("{}", serde_json::to_string(&ask)?);
                            } else {
                                println!(
                                    "[NEW ASK] {} — max {} USDC, {} bid(s) wanted, timeout {}ms",
                                    ask.id,
                                    ask.max_price_per_bid as f64 / 1_000_000.0,
                                    ask.num_bids_wanted,
                                    ask.timeout_ms
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

fn print_asks_table(asks: &[types::object::Object]) {
    if asks.is_empty() {
        println!("No open asks found.");
        return;
    }
    println!("{:<44} {:<12} {:<8} {:<10}", "ASK ID", "MAX PRICE", "BIDS", "STATUS");
    println!("{}", "-".repeat(78));
    for obj in asks {
        if let Some(ask) = obj.deserialize_contents::<types::ask::Ask>(types::object::ObjectType::Ask) {
            println!(
                "{:<44} {:<12.6} {:<8} {:<10}",
                ask.id,
                ask.max_price_per_bid as f64 / 1_000_000.0,
                format!("{}/{}", ask.accepted_bid_count, ask.num_bids_wanted),
                format!("{:?}", ask.status),
            );
        }
    }
}

fn print_asks_json(asks: &[types::object::Object]) {
    let items: Vec<_> = asks
        .iter()
        .filter_map(|obj| {
            obj.deserialize_contents::<types::ask::Ask>(types::object::ObjectType::Ask)
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&items).unwrap_or_default());
}
