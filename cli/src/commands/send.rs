// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
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

    // Get coin reference
    let coin_ref = match coin {
        Some(coin_id) => {
            let obj = client
                .get_object(coin_id)
                .await
                .map_err(|e| anyhow!("Failed to get coin: {}", e.message()))?;
            obj.compute_object_reference()
        }
        None => context
            .get_one_gas_object_owned_by_address(sender)
            .await?
            .ok_or_else(|| anyhow!("No coins found for address {}", sender))?,
    };

    let kind = TransactionKind::TransferCoin { coin: coin_ref, amount: Some(amount), recipient };

    // Use the coin itself as gas
    crate::client_commands::execute_or_serialize(context, sender, kind, Some(coin_ref), tx_args)
        .await
}
