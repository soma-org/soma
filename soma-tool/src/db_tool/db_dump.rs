// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use clap::{Parser, ValueEnum};
use types::committee::EpochId;

use authority::authority_per_epoch_store::AuthorityEpochTables;
use authority::authority_store_pruner::PrunerWatermarks;
use authority::authority_store_tables::AuthorityPerpetualTables;
use authority::checkpoints::{CheckpointStore, CheckpointStoreTables};
use types::storage::committee_store::CommitteeStoreTables;

/// Which logical store to target
#[derive(Clone, Debug, ValueEnum)]
pub enum StoreName {
    /// Perpetual validator tables (objects, effects, etc.)
    Validator,
    /// RPC index tables (owner, balance, target indices)
    Index,
    /// Epoch/committee tables
    Epoch,
    /// Checkpoint tables
    Checkpoint,
}

#[derive(Parser, Clone)]
pub struct DumpOptions {
    /// Which store to inspect
    #[arg(short, long, value_enum)]
    pub store: StoreName,

    /// Epoch number (required for epoch-scoped tables)
    #[arg(short, long)]
    pub epoch: Option<EpochId>,

    /// The table name to inspect
    #[arg(short, long)]
    pub table_name: String,

    /// Page size for paginated output
    #[arg(short, long, default_value = "20")]
    pub page_size: u16,

    /// Page number (0-indexed) for paginated output
    #[arg(short = 'n', long, default_value = "0")]
    pub page_number: usize,
}

/// Convert eyre::Result to anyhow::Result
fn eyre_to_anyhow<T>(result: eyre::Result<T>) -> anyhow::Result<T> {
    result.map_err(|e| anyhow!("{:#}", e))
}

/// List all column families across all known stores
pub fn list_tables(db_path: PathBuf) -> anyhow::Result<()> {
    println!("=== Validator (Perpetual) Tables ===");
    let perpetual_tables = AuthorityPerpetualTables::describe_tables();
    for (name, (key_type, value_type)) in &perpetual_tables {
        println!("  {}: ({}, {})", name, key_type, value_type);
    }

    println!("\n=== Epoch Tables ===");
    let epoch_tables = AuthorityEpochTables::describe_tables();
    for (name, (key_type, value_type)) in &epoch_tables {
        println!("  {}: ({}, {})", name, key_type, value_type);
    }

    println!("\n=== Checkpoint Tables ===");
    let checkpoint_tables = CheckpointStoreTables::describe_tables();
    for (name, (key_type, value_type)) in &checkpoint_tables {
        println!("  {}: ({}, {})", name, key_type, value_type);
    }

    println!("\n=== Committee Tables ===");
    let committee_tables = CommitteeStoreTables::describe_tables();
    for (name, (key_type, value_type)) in &committee_tables {
        println!("  {}: ({}, {})", name, key_type, value_type);
    }

    println!("\n=== Index Tables ===");
    let index_tables = authority::rpc_index::IndexStoreTables::describe_tables();
    for (name, (key_type, value_type)) in &index_tables {
        println!("  {}: ({}, {})", name, key_type, value_type);
    }

    Ok(())
}

/// Dump the contents of a table with pagination
pub fn dump_table(
    store: StoreName,
    epoch: Option<EpochId>,
    db_path: PathBuf,
    table_name: &str,
    page_size: u16,
    page_number: usize,
) -> anyhow::Result<()> {
    match store {
        StoreName::Validator => {
            let db = AuthorityPerpetualTables::open_readonly(&db_path);
            let entries = eyre_to_anyhow(db.dump(table_name, page_size, page_number))?;
            print_entries(table_name, &entries, page_size, page_number);
        }
        StoreName::Epoch => {
            let epoch =
                epoch.ok_or_else(|| anyhow!("--epoch is required for epoch-scoped tables"))?;
            let db = AuthorityEpochTables::open_readonly(epoch, &db_path);
            let entries = eyre_to_anyhow(db.dump(table_name, page_size, page_number))?;
            print_entries(table_name, &entries, page_size, page_number);
        }
        StoreName::Checkpoint => {
            let db = CheckpointStoreTables::open_readonly(&db_path.join("checkpoints"));
            let entries = eyre_to_anyhow(db.dump(table_name, page_size, page_number))?;
            print_entries(table_name, &entries, page_size, page_number);
        }
        StoreName::Index => {
            let db =
                authority::rpc_index::IndexStoreTables::open_readonly(&db_path.join("rpc-index"));
            let entries = eyre_to_anyhow(db.dump(table_name, page_size, page_number))?;
            print_entries(table_name, &entries, page_size, page_number);
        }
    }
    Ok(())
}

/// Show a summary of a table (count of entries via iteration)
pub fn table_summary(
    store: StoreName,
    epoch: Option<EpochId>,
    db_path: PathBuf,
    table_name: &str,
) -> anyhow::Result<()> {
    println!("Table summary for '{}' (iterating to count entries):", table_name);

    let page_size = u16::MAX;
    let entries = match store {
        StoreName::Validator => {
            let db = AuthorityPerpetualTables::open_readonly(&db_path);
            eyre_to_anyhow(db.dump(table_name, page_size, 0))?
        }
        StoreName::Epoch => {
            let epoch =
                epoch.ok_or_else(|| anyhow!("--epoch is required for epoch-scoped tables"))?;
            let db = AuthorityEpochTables::open_readonly(epoch, &db_path);
            eyre_to_anyhow(db.dump(table_name, page_size, 0))?
        }
        StoreName::Checkpoint => {
            let db = CheckpointStoreTables::open_readonly(&db_path.join("checkpoints"));
            eyre_to_anyhow(db.dump(table_name, page_size, 0))?
        }
        StoreName::Index => {
            let db =
                authority::rpc_index::IndexStoreTables::open_readonly(&db_path.join("rpc-index"));
            eyre_to_anyhow(db.dump(table_name, page_size, 0))?
        }
    };

    println!("  Entries (first {} max): {}", page_size, entries.len());
    if !entries.is_empty() {
        if let Some((first_key, _)) = entries.iter().next() {
            println!("  First key: {}", first_key);
        }
        if let Some((last_key, _)) = entries.iter().next_back() {
            println!("  Last key: {}", last_key);
        }
    }

    Ok(())
}

/// Print live file metadata for a store's table
pub fn print_table_metadata(
    store: StoreName,
    epoch: Option<EpochId>,
    db_path: PathBuf,
    table_name: &str,
) -> anyhow::Result<()> {
    println!("DB metadata for table '{}' in {:?} store:", table_name, store);

    match store {
        StoreName::Validator => {
            let _db = AuthorityPerpetualTables::open(&db_path, None);
            let tables = AuthorityPerpetualTables::describe_tables();
            if tables.contains_key(table_name) {
                println!("  Table '{}' exists in validator store", table_name);
            } else {
                println!("  Table '{}' not found in validator store", table_name);
                println!("  Available tables: {:?}", tables.keys().collect::<Vec<_>>());
            }
        }
        StoreName::Epoch => {
            let epoch =
                epoch.ok_or_else(|| anyhow!("--epoch is required for epoch-scoped tables"))?;
            let tables = AuthorityEpochTables::describe_tables();
            if tables.contains_key(table_name) {
                println!("  Table '{}' exists in epoch {} store", table_name, epoch);
            } else {
                println!("  Table '{}' not found in epoch store", table_name);
                println!("  Available tables: {:?}", tables.keys().collect::<Vec<_>>());
            }
        }
        StoreName::Checkpoint => {
            let tables = CheckpointStoreTables::describe_tables();
            if tables.contains_key(table_name) {
                println!("  Table '{}' exists in checkpoint store", table_name);
            } else {
                println!("  Table '{}' not found in checkpoint store", table_name);
                println!("  Available tables: {:?}", tables.keys().collect::<Vec<_>>());
            }
        }
        StoreName::Index => {
            let tables = authority::rpc_index::IndexStoreTables::describe_tables();
            if tables.contains_key(table_name) {
                println!("  Table '{}' exists in index store", table_name);
            } else {
                println!("  Table '{}' not found in index store", table_name);
                println!("  Available tables: {:?}", tables.keys().collect::<Vec<_>>());
            }
        }
    }

    Ok(())
}

fn print_entries(
    table_name: &str,
    entries: &std::collections::BTreeMap<String, String>,
    page_size: u16,
    page_number: usize,
) {
    println!("Table: {} (page {}, page_size {})", table_name, page_number, page_size);
    println!("Entries: {}", entries.len());
    for (key, value) in entries {
        println!("  {} => {}", key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_dump_population() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path();

        // Open all table structs in read-write mode to create them
        let _perpetual = AuthorityPerpetualTables::open(path, None);
        let _epoch = AuthorityEpochTables::open(0, path, None);
        let _checkpoint =
            CheckpointStore::new(&path.join("checkpoints"), Arc::new(PrunerWatermarks::default()));
        let _committee =
            CommitteeStoreTables::open_tables_read_write(path.join("committee"), None, None);

        // Verify describe_tables returns non-empty for all stores
        let perpetual_tables = AuthorityPerpetualTables::describe_tables();
        assert!(!perpetual_tables.is_empty(), "Perpetual tables should not be empty");

        let epoch_tables = AuthorityEpochTables::describe_tables();
        assert!(!epoch_tables.is_empty(), "Epoch tables should not be empty");

        let checkpoint_tables = CheckpointStoreTables::describe_tables();
        assert!(!checkpoint_tables.is_empty(), "Checkpoint tables should not be empty");

        let committee_tables = CommitteeStoreTables::describe_tables();
        assert!(!committee_tables.is_empty(), "Committee tables should not be empty");

        // Verify dump works on each perpetual table (should be empty but not error)
        let perpetual_ro = AuthorityPerpetualTables::open_readonly(path);
        for table_name in perpetual_tables.keys() {
            let result = perpetual_ro.dump(table_name, 10, 0);
            assert!(
                result.is_ok(),
                "Failed to dump perpetual table '{}': {:?}",
                table_name,
                result.err()
            );
        }

        // Verify dump works on each epoch table
        let epoch_ro = AuthorityEpochTables::open_readonly(0, path);
        for table_name in epoch_tables.keys() {
            let result = epoch_ro.dump(table_name, 10, 0);
            assert!(
                result.is_ok(),
                "Failed to dump epoch table '{}': {:?}",
                table_name,
                result.err()
            );
        }

        // Verify dump works on each checkpoint table
        let checkpoint_ro = CheckpointStoreTables::open_readonly(&path.join("checkpoints"));
        for table_name in checkpoint_tables.keys() {
            let result = checkpoint_ro.dump(table_name, 10, 0);
            assert!(
                result.is_ok(),
                "Failed to dump checkpoint table '{}': {:?}",
                table_name,
                result.err()
            );
        }
    }
}
