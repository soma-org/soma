// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use sdk::wallet_context::WalletContext;
use types::object::ObjectType;
use types::settlement::{SellerRating, Settlement};

use crate::response::ClientCommandResponse;

/// Query and display settlement history.
///
/// Requires at least one of --as buyer, --as seller, or an explicit --address.
/// Default: shows settlements for the active wallet address as both buyer and seller.
pub async fn execute(
    context: &mut WalletContext,
    role: Option<String>,
    address: Option<String>,
    limit: u32,
    json: bool,
) -> Result<ClientCommandResponse> {
    let client = context.get_client().await?;

    let addr = match address {
        Some(ref a) => a
            .parse::<types::base::SomaAddress>()
            .map_err(|_| anyhow!("Invalid address: {}", a))?,
        None => context.active_address()?,
    };

    let role = role.as_deref().unwrap_or("both");

    let settlements = match role {
        "buyer" => client
            .get_settlements(Some(&addr), None, Some(limit))
            .await
            .map_err(|e| anyhow!("Failed to get settlements: {}", e.message()))?,
        "seller" => client
            .get_settlements(None, Some(&addr), Some(limit))
            .await
            .map_err(|e| anyhow!("Failed to get settlements: {}", e.message()))?,
        "both" | _ => {
            // Fetch as buyer and seller, dedup by ID
            let mut as_buyer = client
                .get_settlements(Some(&addr), None, Some(limit))
                .await
                .map_err(|e| anyhow!("Failed to get settlements (buyer): {}", e.message()))?;
            let as_seller = client
                .get_settlements(None, Some(&addr), Some(limit))
                .await
                .map_err(|e| anyhow!("Failed to get settlements (seller): {}", e.message()))?;

            let buyer_ids: std::collections::HashSet<_> =
                as_buyer.iter().map(|o| o.id()).collect();
            for obj in as_seller {
                if !buyer_ids.contains(&obj.id()) {
                    as_buyer.push(obj);
                }
            }
            as_buyer
        }
    };

    if json {
        print_settlements_json(&settlements);
    } else {
        print_settlements_table(&settlements);
    }

    Ok(ClientCommandResponse::NoOutput)
}

fn print_settlements_table(settlements: &[types::object::Object]) {
    if settlements.is_empty() {
        println!("No settlements found.");
        return;
    }
    println!(
        "{:<44} {:<44} {:<44} {:<12} {:<10}",
        "SETTLEMENT ID", "BUYER", "SELLER", "AMOUNT", "RATING"
    );
    println!("{}", "-".repeat(148));
    for obj in settlements {
        if let Some(s) = obj.deserialize_contents::<Settlement>(ObjectType::Settlement) {
            let rating = match s.seller_rating {
                SellerRating::Positive => "Positive",
                SellerRating::Negative => "Negative",
            };
            println!(
                "{:<44} {:<44} {:<44} {:<12.6} {:<10}",
                s.id,
                s.buyer,
                s.seller,
                s.amount as f64 / 1_000_000.0,
                rating,
            );
        }
    }
}

fn print_settlements_json(settlements: &[types::object::Object]) {
    let items: Vec<_> = settlements
        .iter()
        .filter_map(|obj| obj.deserialize_contents::<Settlement>(ObjectType::Settlement))
        .collect();
    println!("{}", serde_json::to_string_pretty(&items).unwrap_or_default());
}
