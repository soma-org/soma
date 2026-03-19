// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, bail, ensure};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use types::base::SomaAddress;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Execute the pay command (pay SOMA to multiple recipients)
pub async fn execute(
    context: &mut WalletContext,
    recipients: Vec<KeyIdentity>,
    amounts: Vec<u64>,
    coins: Option<Vec<ObjectID>>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    ensure!(!recipients.is_empty(), "At least one recipient is required");
    ensure!(
        recipients.len() == amounts.len(),
        "Number of recipients ({}) must match number of amounts ({})",
        recipients.len(),
        amounts.len()
    );

    let sender = context.active_address()?;
    let client = context.get_client().await?;

    // Resolve recipient addresses
    let recipient_addresses: Vec<SomaAddress> = recipients
        .into_iter()
        .map(|r| context.get_identity_address(Some(r)))
        .collect::<Result<Vec<_>>>()?;

    // Get coin references and gas payment in a single fetch.
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
            let gas = vec![refs[0]];
            (refs, gas)
        }
        None => {
            // Fetch all coins once. They serve as both the pay-coins input
            // and the gas_payment, so smash_gas can merge dust.
            let all_coins = context.get_gas_objects_sorted_by_balance(sender).await?;
            if all_coins.is_empty() {
                bail!("No coins found for address {}", sender);
            }
            (all_coins.clone(), all_coins)
        }
    };

    let kind = TransactionKind::PayCoins {
        coins: coin_refs,
        amounts: Some(amounts),
        recipients: recipient_addresses,
    };

    crate::client_commands::execute_or_serialize(context, sender, kind, gas_payment, tx_args).await
}
