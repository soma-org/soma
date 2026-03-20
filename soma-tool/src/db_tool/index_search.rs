// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use anyhow::anyhow;
use clap::Parser;
use store::Map as _;
use types::base::SomaAddress;

use authority::rpc_index::{BalanceKey, IndexStoreTables, IndexStoreTablesReadOnly};

#[derive(Parser, Clone)]
pub struct OwnerSearchOptions {
    /// Owner address (hex string)
    #[arg(long)]
    pub owner: String,

    /// Optional object type filter (substring match)
    #[arg(long)]
    pub object_type: Option<String>,

    /// Maximum number of results to return
    #[arg(long, default_value = "20")]
    pub count: u64,
}

#[derive(Parser, Clone)]
pub struct BalanceSearchOptions {
    /// Owner address (hex string)
    #[arg(long)]
    pub owner: String,
}

#[derive(Parser, Clone)]
pub struct TargetSearchOptions {
    /// Optional status filter (open, filled, claimed)
    #[arg(long)]
    pub status: Option<String>,

    /// Optional epoch filter
    #[arg(long)]
    pub epoch: Option<u64>,

    /// Maximum number of results to return
    #[arg(long, default_value = "20")]
    pub count: u64,
}

fn hex_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    let trimmed = s.trim_start_matches("0x");
    if trimmed.len() % 2 != 0 {
        return Err(anyhow!("Hex string must have even length"));
    }
    (0..trimmed.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&trimmed[i..i + 2], 16)
                .map_err(|e| anyhow!("Invalid hex: {}", e))
        })
        .collect()
}

fn parse_address(s: &str) -> anyhow::Result<SomaAddress> {
    let bytes = hex_decode(s)?;
    SomaAddress::try_from(bytes.as_slice()).map_err(|e| anyhow!("Invalid address: {}", e))
}

fn open_index_readonly(db_path: &Path) -> IndexStoreTablesReadOnly {
    let index_path = db_path.join("rpc-index");
    IndexStoreTables::open_readonly(&index_path)
}

/// Search the owner index by address with optional type filter
pub fn search_owner_index(db_path: PathBuf, opts: OwnerSearchOptions) -> anyhow::Result<()> {
    let owner = parse_address(&opts.owner)?;
    let db = open_index_readonly(&db_path);

    let mut count = 0u64;
    let max_count = opts.count;

    for entry in db.owner.safe_iter() {
        let (key, value) = entry?;
        if key.owner != owner {
            if count > 0 {
                break;
            }
            continue;
        }

        if let Some(ref type_filter) = opts.object_type {
            let type_str = format!("{:?}", key.object_type);
            if !type_str.contains(type_filter.as_str()) {
                continue;
            }
        }

        println!(
            "Object: {} | Type: {:?} | Version: {:?} | Balance: {:?}",
            key.object_id, key.object_type, value.version, key.inverted_balance,
        );

        count += 1;
        if count >= max_count {
            break;
        }
    }

    if count == 0 {
        println!("No objects found for owner {}", opts.owner);
    } else {
        println!("\nTotal: {} entries (limit: {})", count, max_count);
    }

    Ok(())
}

/// Look up balance for an address
pub fn search_balance_index(db_path: PathBuf, opts: BalanceSearchOptions) -> anyhow::Result<()> {
    let owner = parse_address(&opts.owner)?;
    let db = open_index_readonly(&db_path);

    let key = BalanceKey { owner };
    match db.balance.get(&key)? {
        Some(info) => {
            println!("Owner: {}", opts.owner);
            println!("Balance delta: {}", info.balance_delta);
            let balance: types::storage::read_store::BalanceInfo = info.into();
            println!("Balance: {}", balance.balance);
        }
        None => {
            println!("No balance entry found for {}", opts.owner);
        }
    }

    Ok(())
}

/// Search the target index by status and/or epoch
pub fn search_target_index(db_path: PathBuf, opts: TargetSearchOptions) -> anyhow::Result<()> {
    let db = open_index_readonly(&db_path);

    let mut count = 0u64;
    let max_count = opts.count;

    for entry in db.targets.safe_iter() {
        let (key, value) = entry?;

        if let Some(ref status) = opts.status {
            if key.status != *status {
                continue;
            }
        }

        if let Some(epoch) = opts.epoch {
            if key.generation_epoch != epoch {
                continue;
            }
        }

        println!(
            "Target: {} | Status: {} | Epoch: {} | Version: {:?}",
            key.target_id, key.status, key.generation_epoch, value.version,
        );

        count += 1;
        if count >= max_count {
            break;
        }
    }

    if count == 0 {
        println!("No targets found");
    } else {
        println!("\nTotal: {} entries (limit: {})", count, max_count);
    }

    Ok(())
}
