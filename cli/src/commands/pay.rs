// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, ensure};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use types::base::SomaAddress;
use types::object::CoinType;
use types::transaction::{BalanceTransferArgs, TransactionKind};

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Execute the pay command (pay SOMA to multiple recipients).
///
/// Stage 13b: balance-mode only. Sender's accumulator is debited
/// directly; no coin object inputs required.
pub async fn execute(
    context: &mut WalletContext,
    recipients: Vec<KeyIdentity>,
    amounts: Vec<u64>,
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

    let recipient_addresses: Vec<SomaAddress> = recipients
        .into_iter()
        .map(|r| context.get_identity_address(Some(r)))
        .collect::<Result<Vec<_>>>()?;

    let transfers: Vec<_> =
        recipient_addresses.into_iter().zip(amounts.into_iter()).collect();

    let kind = TransactionKind::BalanceTransfer(BalanceTransferArgs {
        coin_type: CoinType::Soma,
        transfers,
    });

    crate::client_commands::execute_or_serialize(context, sender, kind, vec![], tx_args).await
}
