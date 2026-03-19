// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::{ClientCommandResponse, TransactionResponse};

/// Execute the send command (transfer SOMA to a recipient)
pub async fn execute(
    context: &mut WalletContext,
    to: KeyIdentity,
    amount: u64,
    coin: Option<ObjectID>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;
    let recipient = context.get_identity_address(Some(to))?;
    let client = context.get_client().await?;

    // Get coin reference and gas payment coins in a single fetch.
    let (coin_ref, gas_payment) = match coin {
        Some(coin_id) => {
            let obj = client
                .get_object(coin_id)
                .await
                .map_err(|e| anyhow!("Failed to get coin: {}", e.message()))?;
            let r = obj.compute_object_reference();
            (r, vec![r])
        }
        None => {
            // Fetch all coins once. The richest (first) is used for the
            // transfer; ALL of them are passed as gas_payment so smash_gas
            // can merge dust coins, keeping the address clean.
            let coins = context.get_gas_objects_sorted_by_balance(sender).await?;
            let r =
                *coins.first().ok_or_else(|| anyhow!("No coins found for address {}", sender))?;
            (r, coins)
        }
    };

    let kind = TransactionKind::TransferCoin { coin: coin_ref, amount: Some(amount), recipient };

    crate::client_commands::execute_or_serialize(context, sender, kind, gas_payment, tx_args).await
}
