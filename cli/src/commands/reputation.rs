// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use sdk::wallet_context::WalletContext;
use serde::Serialize;
use types::base::SomaAddress;

use crate::response::ClientCommandResponse;

#[derive(Debug, Serialize)]
pub struct ReputationSummary {
    pub address: String,

    // As buyer
    pub buyer_settlements: u64,
    pub buyer_volume_spent_usdc: f64,
    pub buyer_unique_sellers: u64,

    // As seller
    pub seller_settlements: u64,
    pub seller_volume_earned_usdc: f64,
    pub seller_negative_ratings: u64,
    pub seller_approval_rate: Option<f64>,
    pub seller_unique_buyers: u64,
}

/// Query and display reputation for an address.
///
/// Uses the GetReputation RPC endpoint which computes reputation server-side
/// from settlement indexes.
pub async fn execute(
    context: &mut WalletContext,
    address: Option<String>,
    json: bool,
) -> Result<ClientCommandResponse> {
    let client = context.get_client().await?;

    let addr: SomaAddress = match address {
        Some(ref a) => a
            .parse()
            .map_err(|_| anyhow!("Invalid address: {}", a))?,
        None => context.active_address()?,
    };

    let resp = client
        .get_reputation(&addr)
        .await
        .map_err(|e| anyhow!("Failed to get reputation: {}", e.message()))?;

    let buyer_volume = resp.buyer_volume_spent.unwrap_or(0);
    let seller_volume = resp.seller_volume_earned.unwrap_or(0);

    let summary = ReputationSummary {
        address: addr.to_string(),
        buyer_settlements: resp.buyer_settlements.unwrap_or(0),
        buyer_volume_spent_usdc: buyer_volume as f64 / 1_000_000.0,
        buyer_unique_sellers: resp.buyer_unique_sellers.unwrap_or(0),
        seller_settlements: resp.seller_settlements.unwrap_or(0),
        seller_volume_earned_usdc: seller_volume as f64 / 1_000_000.0,
        seller_negative_ratings: resp.seller_negative_ratings.unwrap_or(0),
        seller_approval_rate: resp.seller_approval_rate,
        seller_unique_buyers: resp.seller_unique_buyers.unwrap_or(0),
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("Reputation for {}", addr);
        println!();
        println!("  As Buyer:");
        println!("    Settlements:     {}", summary.buyer_settlements);
        println!("    Volume Spent:    {:.6} USDC", summary.buyer_volume_spent_usdc);
        println!("    Unique Sellers:  {}", summary.buyer_unique_sellers);
        println!();
        println!("  As Seller:");
        println!("    Settlements:     {}", summary.seller_settlements);
        println!("    Volume Earned:   {:.6} USDC", summary.seller_volume_earned_usdc);
        println!(
            "    Approval Rate:   {}",
            match summary.seller_approval_rate {
                Some(rate) => format!("{:.1}%", rate),
                None => "N/A (no settlements)".to_string(),
            }
        );
        println!("    Negative Ratings: {}", summary.seller_negative_ratings);
        println!("    Unique Buyers:   {}", summary.seller_unique_buyers);
    }

    Ok(ClientCommandResponse::NoOutput)
}
