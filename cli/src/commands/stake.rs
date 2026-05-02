// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use sdk::wallet_context::WalletContext;
use types::base::SomaAddress;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::TxProcessingArgs;
use crate::response::ClientCommandResponse;

/// Execute the stake command (stake SOMA with a validator)
pub async fn execute_stake(
    context: &mut WalletContext,
    validator: SomaAddress,
    amount: Option<u64>,
    coin: Option<ObjectID>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;
    let client = context.get_client().await?;

    // Get coin reference and gas payment in a single fetch.
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
            let (r, balance) = context
                .get_richest_coin_with_balance(sender)
                .await?
                .ok_or_else(|| anyhow!("No coins found for address {}", sender))?;
            if let Some(amt) = amount {
                if balance < amt {
                    return Err(anyhow!(
                        "Richest coin has balance {} but stake requires {}. \
                         Run `soma merge-coins` to consolidate your coins.",
                        balance,
                        amt,
                    ));
                }
            }
            (r, vec![r])
        }
    };

    let kind = TransactionKind::AddStake { address: validator, coin_ref, amount };

    crate::client_commands::execute_or_serialize(context, sender, kind, gas_payment, tx_args).await
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

    crate::client_commands::execute_or_serialize(context, sender, kind, vec![], tx_args).await
}

/// Stage 9d: list a staker's active delegations using the new
/// `ListDelegations` RPC endpoint. Replaces the equivalent
/// owned-StakedSomaV1-object scan path; eventually the only path once
/// Stage 9d full-removal lands.
pub async fn execute_list_stakes(
    context: &mut WalletContext,
    staker: SomaAddress,
    json: bool,
) -> Result<()> {
    let client = context.get_client().await?;

    let request = rpc::proto::soma::ListDelegationsRequest::default()
        .with_staker(staker.to_string());
    let response = client
        .list_delegations(request)
        .await
        .map_err(|e| anyhow!("ListDelegations RPC failed: {}", e.message()))?;

    if json {
        let rows: Vec<_> = response
            .delegations
            .iter()
            .map(|d| {
                serde_json::json!({
                    "pool_id": d.pool_id,
                    "activation_epoch": d.activation_epoch,
                    "principal": d.principal,
                })
            })
            .collect();
        let payload = serde_json::json!({
            "staker": staker.to_string(),
            "total_principal": response.total_principal.unwrap_or(0),
            "delegations": rows,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if response.delegations.is_empty() {
        println!("No active stakes for {}", staker);
    } else {
        println!("Stakes for {}:", staker);
        println!(
            "  {:<66}  {:<16}  {}",
            "POOL", "ACTIVATION_EPOCH", "PRINCIPAL"
        );
        for d in &response.delegations {
            println!(
                "  {:<66}  {:<16}  {}",
                d.pool_id.as_deref().unwrap_or(""),
                d.activation_epoch.unwrap_or(0),
                d.principal.unwrap_or(0),
            );
        }
        println!(
            "Total principal: {} shannons",
            response.total_principal.unwrap_or(0)
        );
    }
    Ok(())
}
