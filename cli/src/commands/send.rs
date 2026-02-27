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

    // Get coin reference and gas strategy.
    // When auto-selecting: pick the richest coin for the transfer and pass
    // gas=None so the TransactionBuilder includes ALL coins as gas_payment.
    // smash_gas will then merge any dust coins into the primary, keeping the
    // address clean.
    let (coin_ref, explicit_gas) = match coin {
        Some(coin_id) => {
            let obj = client
                .get_object(coin_id)
                .await
                .map_err(|e| anyhow!("Failed to get coin: {}", e.message()))?;
            let r = obj.compute_object_reference();
            (r, Some(r))
        }
        None => {
            let r = context
                .get_richest_gas_object_owned_by_address(sender)
                .await?
                .ok_or_else(|| anyhow!("No coins found for address {}", sender))?;
            (r, None) // None → TransactionBuilder auto-selects all coins for gas
        }
    };

    let kind = TransactionKind::TransferCoin { coin: coin_ref, amount: Some(amount), recipient };

    crate::client_commands::execute_or_serialize(context, sender, kind, explicit_gas, tx_args).await
}
