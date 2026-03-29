// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, ensure};
use sdk::wallet_context::WalletContext;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Accept a bid by ID (atomic settlement: deducts USDC, credits seller vault, creates settlement).
///
/// The ask-id is inferred from the bid object on-chain when not provided.
/// The payment coin is auto-selected from the sender's richest USDC coin when not provided.
pub async fn execute(
    context: &mut WalletContext,
    bid_id: ObjectID,
    ask_id: Option<ObjectID>,
    payment_coin: Option<ObjectID>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;
    let client = context.get_client().await?;

    // Look up the bid object to infer ask_id if not provided
    let ask_id = match ask_id {
        Some(id) => id,
        None => {
            let bid_obj = client
                .get_object(bid_id)
                .await
                .map_err(|e| anyhow!("Failed to get bid object: {}", e.message()))?;
            let bid: types::bid::Bid = bid_obj
                .deserialize_contents(types::object::ObjectType::Bid)
                .ok_or_else(|| anyhow!("Failed to deserialize bid object {}", bid_id))?;
            bid.ask_id
        }
    };

    // Get payment coin (must be USDC) — auto-select if not provided
    let payment_ref = match payment_coin {
        Some(coin_id) => {
            let obj = client
                .get_object(coin_id)
                .await
                .map_err(|e| anyhow!("Failed to get payment coin: {}", e.message()))?;
            obj.compute_object_reference()
        }
        None => {
            let (usdc_ref, _) = context
                .get_richest_usdc_coin(sender)
                .await?
                .ok_or_else(|| {
                    anyhow!(
                        "No USDC coins found for address {}. Bridge USDC first, \
                         or specify --payment-coin explicitly.",
                        sender
                    )
                })?;
            usdc_ref
        }
    };

    let kind = TransactionKind::AcceptBid(types::transaction::AcceptBidArgs {
        ask_id,
        bid_id,
        payment_coin: payment_ref,
    });

    let (gas_ref, _) = context
        .get_richest_coin_with_balance(sender)
        .await?
        .ok_or_else(|| anyhow!("No SOMA coins found for gas payment for address {}", sender))?;
    crate::client_commands::execute_or_serialize(context, sender, kind, vec![gas_ref], tx_args)
        .await
}

/// Accept the cheapest pending bid(s) for an ask.
///
/// Fetches all bids for the ask, sorts by price ascending, and accepts up to `count` cheapest.
/// Currently requires an RPC endpoint or object scan to list bids — this scans owned objects.
pub async fn execute_cheapest(
    context: &mut WalletContext,
    ask_id: ObjectID,
    count: u32,
    payment_coin: Option<ObjectID>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    ensure!(count > 0, "Count must be at least 1");

    let sender = context.active_address()?;
    let client = context.get_client().await?;

    // Fetch the ask to verify it exists and is open
    let ask_obj = client
        .get_object(ask_id)
        .await
        .map_err(|e| anyhow!("Failed to get ask object: {}", e.message()))?;
    let ask: types::ask::Ask = ask_obj
        .deserialize_contents(types::object::ObjectType::Ask)
        .ok_or_else(|| anyhow!("Failed to deserialize ask object {}", ask_id))?;

    ensure!(
        ask.status == types::ask::AskStatus::Open,
        "Ask {} is not open (status: {:?})",
        ask_id,
        ask.status
    );

    let remaining = ask.num_bids_wanted.saturating_sub(ask.accepted_bid_count);
    ensure!(remaining > 0, "Ask {} is already filled", ask_id);

    let accept_count = count.min(remaining);

    // Fetch all pending bids for this ask via the secondary index
    let bid_objects = client
        .get_bids_for_ask(ask_id, Some("Pending"))
        .await
        .map_err(|e| anyhow!("Failed to get bids for ask: {}", e.message()))?;

    ensure!(
        !bid_objects.is_empty(),
        "No pending bids found for ask {}",
        ask_id
    );

    // Deserialize bids and sort by price ascending (cheapest first)
    let mut bids: Vec<(types::bid::Bid, types::object::ObjectRef)> = bid_objects
        .iter()
        .filter_map(|obj| {
            let bid = obj.deserialize_contents::<types::bid::Bid>(types::object::ObjectType::Bid)?;
            Some((bid, obj.compute_object_reference()))
        })
        .collect();

    bids.sort_by_key(|(bid, _)| bid.price);

    ensure!(
        !bids.is_empty(),
        "No pending bids could be deserialized for ask {}",
        ask_id
    );

    let to_accept = &bids[..accept_count.min(bids.len() as u32) as usize];

    // Accept each bid sequentially (each AcceptBid mutates the ask's accepted_bid_count)
    let mut last_response = None;
    for (i, (bid, _bid_ref)) in to_accept.iter().enumerate() {
        // Get fresh payment coin for each accept (balance changes after each)
        let payment_ref = match payment_coin {
            Some(coin_id) => {
                let obj = client
                    .get_object(coin_id)
                    .await
                    .map_err(|e| anyhow!("Failed to get payment coin: {}", e.message()))?;
                obj.compute_object_reference()
            }
            None => {
                let (usdc_ref, _) = context
                    .get_richest_usdc_coin(sender)
                    .await?
                    .ok_or_else(|| {
                        anyhow!(
                            "No USDC coins found for address {}. Bridge USDC first.",
                            sender
                        )
                    })?;
                usdc_ref
            }
        };

        let kind = TransactionKind::AcceptBid(types::transaction::AcceptBidArgs {
            ask_id,
            bid_id: bid.id,
            payment_coin: payment_ref,
        });

        let (gas_ref, _) = context
            .get_richest_coin_with_balance(sender)
            .await?
            .ok_or_else(|| anyhow!("No SOMA coins for gas (accept {} of {})", i + 1, to_accept.len()))?;

        let response = crate::client_commands::execute_or_serialize(
            context,
            sender,
            kind,
            vec![gas_ref],
            tx_args.clone(),
        )
        .await?;

        eprintln!(
            "Accepted bid {} ({} of {}) — price: {} USDC",
            bid.id,
            i + 1,
            to_accept.len(),
            bid.price as f64 / 1_000_000.0
        );
        last_response = Some(response);
    }

    last_response.ok_or_else(|| anyhow!("No bids were accepted"))
}
