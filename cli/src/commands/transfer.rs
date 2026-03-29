// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, ensure};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Unified coin transfer — sends SOMA or USDC to one or more recipients.
///
/// Replaces the old `send` (single recipient) and `pay` (multi-recipient) commands.
/// Defaults to SOMA; use `usdc: true` to send USDC instead.
pub async fn execute(
    context: &mut WalletContext,
    amount: u64,
    recipients: Vec<KeyIdentity>,
    amounts: Option<Vec<u64>>,
    coins: Option<Vec<ObjectID>>,
    usdc: bool,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    ensure!(!recipients.is_empty(), "At least one recipient is required");

    let sender = context.active_address()?;
    let client = context.get_client().await?;

    // Resolve recipient addresses
    let recipient_addresses: Vec<types::base::SomaAddress> = recipients
        .into_iter()
        .map(|r| context.get_identity_address(Some(r)))
        .collect::<Result<Vec<_>>>()?;

    // Build per-recipient amounts
    let send_amounts = match amounts {
        Some(per_recipient) => {
            ensure!(
                per_recipient.len() == recipient_addresses.len(),
                "Number of --amounts ({}) must match number of recipients ({})",
                per_recipient.len(),
                recipient_addresses.len()
            );
            per_recipient
        }
        None => {
            // Single amount split equally or sent to single recipient
            if recipient_addresses.len() == 1 {
                vec![amount]
            } else {
                // Multiple recipients, single amount — divide equally
                let per_each = amount / recipient_addresses.len() as u64;
                ensure!(per_each > 0, "Amount too small to split among {} recipients", recipient_addresses.len());
                let mut amts = vec![per_each; recipient_addresses.len()];
                // Give remainder to last recipient
                let remainder = amount - per_each * recipient_addresses.len() as u64;
                if remainder > 0 {
                    *amts.last_mut().unwrap() += remainder;
                }
                amts
            }
        }
    };

    let total_needed: u64 = send_amounts.iter().sum();

    // Select coins based on token type
    let (coin_refs, gas_payment) = match coins {
        Some(coin_ids) => {
            ensure!(!coin_ids.is_empty(), "At least one input coin is required");
            let mut refs = Vec::new();
            for coin_id in &coin_ids {
                let coin = client
                    .get_object(*coin_id)
                    .await
                    .map_err(|e| anyhow!("Failed to get coin {}: {}", coin_id, e.message()))?;
                refs.push(coin.compute_object_reference());
            }
            if usdc {
                // USDC transfer: coins are USDC, need separate SOMA for gas
                let (gas_ref, _) = context
                    .get_richest_coin_with_balance(sender)
                    .await?
                    .ok_or_else(|| anyhow!("No SOMA coins for gas. Use `soma faucet` to get gas tokens."))?;
                (refs, vec![gas_ref])
            } else {
                let gas = vec![refs[0]];
                (refs, gas)
            }
        }
        None => {
            if usdc {
                // Auto-select USDC coin for transfer, SOMA for gas
                let (usdc_ref, usdc_balance) = context
                    .get_richest_usdc_coin(sender)
                    .await?
                    .ok_or_else(|| anyhow!("No USDC coins found for address {}. Bridge USDC first.", sender))?;
                if usdc_balance < total_needed {
                    return Err(anyhow!(
                        "Richest USDC coin has {} microdollars but transfer requires {}.",
                        usdc_balance,
                        total_needed,
                    ));
                }
                let (gas_ref, _) = context
                    .get_richest_coin_with_balance(sender)
                    .await?
                    .ok_or_else(|| anyhow!("No SOMA coins for gas. Use `soma faucet` to get gas tokens."))?;
                (vec![usdc_ref], vec![gas_ref])
            } else {
                // Auto-select SOMA coin for both transfer and gas
                let (r, balance) = context
                    .get_richest_coin_with_balance(sender)
                    .await?
                    .ok_or_else(|| anyhow!("No SOMA coins found for address {}", sender))?;
                if balance < total_needed {
                    return Err(anyhow!(
                        "Richest coin has balance {} but transfer requires {}. \
                         Run `soma merge-coins` to consolidate your coins.",
                        balance,
                        total_needed,
                    ));
                }
                (vec![r], vec![r])
            }
        }
    };

    let kind = TransactionKind::Transfer {
        coins: coin_refs,
        amounts: Some(send_amounts),
        recipients: recipient_addresses,
    };

    crate::client_commands::execute_or_serialize(context, sender, kind, gas_payment, tx_args).await
}

/// Transfer an arbitrary object to a recipient (non-coin transfer).
pub async fn execute_transfer_object(
    context: &mut WalletContext,
    to: KeyIdentity,
    object_id: ObjectID,
    gas: Option<ObjectID>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.get_object_owner(&object_id).await?;
    let recipient = context.get_identity_address(Some(to))?;
    let client = context.get_client().await?;

    let object =
        client.get_object(object_id).await.map_err(|e| anyhow!("Failed to get object: {}", e))?;
    let object_ref = object.compute_object_reference();

    let kind = TransactionKind::TransferObjects { objects: vec![object_ref], recipient };

    let gas_payment = match gas {
        Some(gas_id) => {
            let gas_obj = client
                .get_object(gas_id)
                .await
                .map_err(|e| anyhow!("Failed to get gas object: {}", e))?;
            vec![gas_obj.compute_object_reference()]
        }
        None => vec![],
    };

    crate::client_commands::execute_or_serialize(context, sender, kind, gas_payment, tx_args).await
}
