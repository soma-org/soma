// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use sdk::wallet_context::WalletContext;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Submit a negative seller rating on a settlement.
///
/// Only negative ratings go on-chain — the default is positive (no tx needed).
/// Must be submitted within the rating deadline.
pub async fn execute(
    context: &mut WalletContext,
    settlement_id: ObjectID,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;

    let kind = TransactionKind::RateSeller { settlement_id };

    let (gas_ref, _) = context
        .get_richest_coin_with_balance(sender)
        .await?
        .ok_or_else(|| anyhow!("No coins found for address {}", sender))?;
    crate::client_commands::execute_or_serialize(context, sender, kind, vec![gas_ref], tx_args)
        .await
}
