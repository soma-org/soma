// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma stake {add, remove, list}` — manage delegations to validators.

use anyhow::{Result, anyhow};
use clap::Parser;
use sdk::wallet_context::WalletContext;
use types::base::SomaAddress;
use types::object::ObjectID;
use types::transaction::TransactionKind;

use crate::client_commands::{TxProcessingArgs, execute_or_serialize};
use crate::response::ClientCommandResponse;
use crate::soma_amount::SomaAmount;

#[derive(Parser, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum StakeCommand {
    /// Stake SOMA with a validator.
    #[clap(after_help = "\
EXAMPLES:
    soma stake add --validator 0xVAL... --amount 10")]
    Add {
        /// Validator address to stake with.
        #[clap(long)]
        validator: SomaAddress,
        /// Amount to stake in SOMA, debited from your SOMA balance.
        #[clap(long)]
        amount: SomaAmount,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Withdraw staked SOMA from a pool. Omit `--amount` to withdraw
    /// the full stake.
    #[clap(after_help = "\
EXAMPLES:
    soma stake remove --pool 0xPOOL_ID
    soma stake remove --pool 0xPOOL_ID --amount 5")]
    Remove {
        /// StakingPool ObjectID. Use `soma stake list` to find yours.
        #[clap(long)]
        pool: ObjectID,
        /// Amount to withdraw in SOMA. Omit to withdraw your full stake.
        #[clap(long)]
        amount: Option<SomaAmount>,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// List active stakes for an address (defaults to the active wallet).
    #[clap(after_help = "\
EXAMPLES:
    soma stake list
    soma stake list --staker 0xADDR...")]
    List {
        /// Staker address (defaults to the active wallet address).
        #[clap(long)]
        staker: Option<SomaAddress>,
    },
}

impl StakeCommand {
    pub async fn execute(
        self,
        context: &mut WalletContext,
        json: bool,
    ) -> Result<ClientCommandResponse> {
        match self {
            Self::Add { validator, amount, tx_args } => {
                let sender = context.active_address()?;
                let amount = amount.shannons();
                if amount == 0 {
                    return Err(anyhow!("Stake amount must be greater than zero"));
                }
                let kind = TransactionKind::AddStake { validator, amount };
                execute_or_serialize(context, sender, kind, tx_args).await
            }

            Self::Remove { pool, amount, tx_args } => {
                let sender = context.active_address()?;
                let kind = TransactionKind::WithdrawStake {
                    pool_id: pool,
                    amount: amount.map(|a| a.shannons()),
                };
                execute_or_serialize(context, sender, kind, tx_args).await
            }

            Self::List { staker } => {
                let staker = match staker {
                    Some(addr) => addr,
                    None => context.active_address()?,
                };
                list_stakes(context, staker, json).await?;
                Ok(ClientCommandResponse::NoOutput)
            }
        }
    }
}

async fn list_stakes(
    context: &mut WalletContext,
    staker: SomaAddress,
    json: bool,
) -> Result<()> {
    let client = context.get_client().await?;

    let request =
        rpc::proto::soma::ListDelegationsRequest::default().with_staker(staker.to_string());
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
            println!("  {:<66}  {}", d.pool_id.as_deref().unwrap_or(""), d.principal.unwrap_or(0));
        }
        println!("Total principal: {} shannons", response.total_principal.unwrap_or(0));
    }
    Ok(())
}
