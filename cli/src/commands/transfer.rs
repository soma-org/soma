// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, ensure};
use sdk::wallet_context::WalletContext;
use soma_keys::key_identity::KeyIdentity;
use types::object::{CoinType, ObjectID};
use types::transaction::{BalanceTransferArgs, TransactionKind};

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Unified balance transfer — sends SOMA or USDC to one or more recipients.
///
/// Stage 13b: balance-mode only. The sender's accumulator is debited
/// directly; no coin object inputs.
pub async fn execute(
    context: &mut WalletContext,
    amount: u64,
    recipients: Vec<KeyIdentity>,
    amounts: Option<Vec<u64>>,
    usdc: bool,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    ensure!(!recipients.is_empty(), "At least one recipient is required");

    let sender = context.active_address()?;

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
            if recipient_addresses.len() == 1 {
                vec![amount]
            } else {
                let per_each = amount / recipient_addresses.len() as u64;
                ensure!(per_each > 0, "Amount too small to split among {} recipients", recipient_addresses.len());
                let mut amts = vec![per_each; recipient_addresses.len()];
                let remainder = amount - per_each * recipient_addresses.len() as u64;
                if remainder > 0 {
                    *amts.last_mut().unwrap() += remainder;
                }
                amts
            }
        }
    };

    let coin_type = if usdc { CoinType::Usdc } else { CoinType::Soma };
    let transfers: Vec<_> = recipient_addresses
        .into_iter()
        .zip(send_amounts.into_iter())
        .collect();

    let kind = TransactionKind::BalanceTransfer(BalanceTransferArgs {
        coin_type,
        transfers,
    });

    crate::client_commands::execute_or_serialize(context, sender, kind, vec![], tx_args).await
}

// Stage 13b: keep `_` to suppress unused-import warning if any
// callers stop passing ObjectID after migrating.
const _UNUSED_OBJECT_ID: Option<ObjectID> = None;

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
