// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, bail};
use sdk::wallet_context::WalletContext;
use types::base::SomaAddress;
use types::model::ModelId;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::{ClientCommandResponse, TransactionResponse};

/// Execute the stake command (stake SOMA with a validator or model)
pub async fn execute_stake(
    context: &mut WalletContext,
    validator: Option<SomaAddress>,
    model: Option<ModelId>,
    amount: Option<u64>,
    coin: Option<ObjectID>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;
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

    // Build transaction kind
    let kind = if let Some(validator_address) = validator {
        TransactionKind::AddStake { address: validator_address, coin_ref, amount }
    } else if let Some(model_id) = model {
        TransactionKind::AddStakeToModel { model_id, coin_ref, amount }
    } else {
        unreachable!()
    };

    crate::client_commands::execute_or_serialize(context, sender, kind, Some(coin_ref), tx_args)
        .await
}

/// Execute the unstake command (withdraw staked SOMA)
pub async fn execute_unstake(
    context: &mut WalletContext,
    staked_soma_id: ObjectID,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;
    let client = context.get_client().await?;

    // Get staked soma reference
    let staked_obj = client
        .get_object(staked_soma_id)
        .await
        .map_err(|e| anyhow!("Failed to get staked SOMA object: {}", e.message()))?;
    let staked_ref = staked_obj.compute_object_reference();

    let kind = TransactionKind::WithdrawStake { staked_soma: staked_ref };

    crate::client_commands::execute_or_serialize(context, sender, kind, None, tx_args).await
}
