// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! `soma merge-coins` (Stage 13b: stub, no-op).
//!
//! Pre-Stage 13 the CLI offered a "merge coins" command that
//! consolidated the address's many small SOMA coin objects into
//! one. Stage 13a removed Coin objects from genesis; Stage 13b
//! deleted the coin-mode TransferKind that the merge implementation
//! relied on. Balance-mode has no concept of "many objects" — the
//! accumulator already holds a single u64 per (owner, coin_type).
//!
//! The command remains in the CLI surface as a no-op so existing
//! scripts don't break, but it always reports "nothing to merge."

use std::fmt::{self, Display, Formatter};

use anyhow::Result;
use colored::Colorize;
use sdk::wallet_context::WalletContext;
use serde::Serialize;

pub async fn execute(_context: &mut WalletContext) -> Result<MergeCoinsResponse> {
    Ok(MergeCoinsResponse::NothingToMerge)
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MergeCoinsResponse {
    NothingToMerge,
}

impl Display for MergeCoinsResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MergeCoinsResponse::NothingToMerge => {
                write!(
                    f,
                    "{}",
                    "[deprecated] `merge-coins` is a no-op: balance-mode has no \
                     coin objects to merge. The accumulator stores a single u64 \
                     per (owner, coin_type), so consolidation is meaningless. \
                     This command will be removed in a future release."
                        .yellow()
                )
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
