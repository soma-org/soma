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
    amount: u64,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;

    if amount == 0 {
        return Err(anyhow!("Stake amount must be greater than zero"));
    }

    // Stage 9d-C2: AddStake is balance-mode — the executor debits
    // `amount` SOMA from the sender's accumulator, no coin reference
    // required. Gas is balance-mode too (vec![]).
    let kind = TransactionKind::AddStake { validator, amount };

    crate::client_commands::execute_or_serialize(context, sender, kind, tx_args).await
}

/// Execute the unstake command (Stage 9d-C3: balance-mode).
/// Pays pending F1 rewards + the requested principal amount to the
/// sender's SOMA balance. `amount = None` drains the entire row.
pub async fn execute_unstake(
    context: &mut WalletContext,
    pool_id: ObjectID,
    amount: Option<u64>,
    tx_args: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    let sender = context.active_address()?;

    let kind = TransactionKind::WithdrawStake { pool_id, amount };

    crate::client_commands::execute_or_serialize(context, sender, kind, tx_args).await
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
                    "principal": d.principal,
                    "last_collected_period": d.last_collected_period,
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
        println!("  {:<66}  {}", "POOL", "PRINCIPAL");
        for d in &response.delegations {
            println!(
                "  {:<66}  {}",
                d.pool_id.as_deref().unwrap_or(""),
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
