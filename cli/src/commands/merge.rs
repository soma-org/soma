// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! Merge coins command (`soma merge-coins`).

use std::fmt::{self, Display, Formatter};

use anyhow::Result;
use colored::Colorize;
use futures::TryStreamExt as _;
use sdk::wallet_context::WalletContext;
use serde::Serialize;
use types::effects::TransactionEffectsAPI as _;
use types::transaction::{TransactionData, TransactionKind};

use crate::response::TransactionResponse;

pub async fn execute(context: &mut WalletContext) -> Result<MergeCoinsResponse> {
    let sender = context.active_address()?;
    let client = context.get_client().await?;

    // 1. List coins (one page — up to 1000, run again if more)
    const MAX_COINS: usize = 1000;
    let mut request = rpc::proto::soma::ListOwnedObjectsRequest::default();
    request.owner = Some(sender.to_string());
    request.page_size = Some(MAX_COINS as u32);
    request.object_type = Some(rpc::types::ObjectType::Coin.into());

    let stream = client.list_owned_objects(request).await;
    tokio::pin!(stream);

    let mut coins: Vec<(types::object::ObjectRef, u64)> = Vec::new();
    while let Some(obj) = stream.try_next().await? {
        let balance = obj.as_coin().unwrap_or(0);
        let obj_ref = obj.compute_object_reference();
        coins.push((obj_ref, balance));
        if coins.len() >= MAX_COINS {
            break;
        }
    }

    if coins.len() <= 1 {
        return Ok(MergeCoinsResponse::NothingToMerge);
    }

    // 2. Sort by balance ascending — smallest first
    coins.sort_by(|a, b| a.1.cmp(&b.1));

    let transfer_coin = coins[0].0;
    let gas_payment: Vec<_> = coins[1..].iter().map(|(r, _)| *r).collect();

    let kind =
        TransactionKind::TransferCoin { coin: transfer_coin, amount: None, recipient: sender };
    let tx_data = TransactionData::new(kind, sender, gas_payment);

    // 3. Sign and execute
    let tx = context.sign_transaction(&tx_data).await;
    let response = context.execute_transaction_may_fail(tx).await?;

    if !response.effects.status().is_ok() {
        anyhow::bail!("MergeCoins failed: {:?}", response.effects.status());
    }

    let tx_response = TransactionResponse::from_effects(
        &response.effects,
        Some(response.checkpoint_sequence_number),
    );
    Ok(MergeCoinsResponse::Success(tx_response))
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MergeCoinsResponse {
    Success(TransactionResponse),
    NothingToMerge,
}

impl Display for MergeCoinsResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MergeCoinsResponse::Success(tx) => {
                writeln!(f, "{}", "Coins merged successfully.".green().bold())?;
                write!(f, "{}", tx)
            }
            MergeCoinsResponse::NothingToMerge => {
                write!(f, "{}", "Nothing to merge: address has 0 or 1 coins.".yellow())
            }
        }
    }
}

impl MergeCoinsResponse {
    pub fn print(&self, json: bool) {
        if json {
            match serde_json::to_string_pretty(self) {
                Ok(s) => println!("{}", s),
                Err(e) => eprintln!("Failed to serialize response: {}", e),
            }
        } else {
            println!("{}", self);
        }
    }
}
