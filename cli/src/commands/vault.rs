// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use clap::Parser;
use sdk::wallet_context::WalletContext;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;
use crate::usdc_amount::UsdcAmount;

/// Subcommands for `soma vault`
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum VaultCommand {
    /// Withdraw USDC from a seller vault
    Withdraw {
        /// The vault object ID
        vault_id: ObjectID,
        /// Amount to withdraw in USDC (default: full balance)
        #[clap(long)]
        amount: Option<UsdcAmount>,
        /// Existing USDC coin to credit (creates new coin if not provided)
        #[clap(long)]
        recipient_coin: Option<ObjectID>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },
}

pub async fn execute(
    context: &mut WalletContext,
    cmd: VaultCommand,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;
    let client = context.get_client().await?;

    match cmd {
        VaultCommand::Withdraw { vault_id, amount, recipient_coin, tx_args } => {
            let vault_obj = client
                .get_object(vault_id)
                .await
                .map_err(|e| anyhow!("Failed to get vault object: {}", e.message()))?;
            let vault_ref = vault_obj.compute_object_reference();

            let recipient_ref = match recipient_coin {
                Some(coin_id) => {
                    let obj = client
                        .get_object(coin_id)
                        .await
                        .map_err(|e| anyhow!("Failed to get recipient coin: {}", e.message()))?;
                    Some(obj.compute_object_reference())
                }
                None => None,
            };

            let kind = TransactionKind::WithdrawFromVault {
                vault: vault_ref,
                amount: amount.map(|a| a.microdollars()),
                recipient_coin: recipient_ref,
            };

            let (gas_ref, _) = context
                .get_richest_coin_with_balance(sender)
                .await?
                .ok_or_else(|| anyhow!("No coins found for address {}", sender))?;
            crate::client_commands::execute_or_serialize(
                context,
                sender,
                kind,
                vec![gas_ref],
                tx_args,
            )
            .await
        }
    }
}
