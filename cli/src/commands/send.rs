// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use types::object::CoinType;
use types::transaction::{BalanceTransferArgs, TransactionKind};

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Execute the send command (transfer SOMA to a recipient).
///
/// Stage 13b: balance-mode only. Debits SOMA directly from the
/// sender's accumulator — no coin reference required.
pub async fn execute(
    context: &mut WalletContext,
    to: KeyIdentity,
    amount: u64,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;
    let recipient = context.get_identity_address(Some(to))?;

    let kind = TransactionKind::BalanceTransfer(BalanceTransferArgs {
        coin_type: CoinType::Soma,
        transfers: vec![(recipient, amount)],
    });

    crate::client_commands::execute_or_serialize(context, sender, kind, vec![], tx_args).await
}
