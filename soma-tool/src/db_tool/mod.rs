// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::anyhow;
use clap::Parser;
use store::Map as _;
use tracing::info;
use types::checkpoints::CheckpointSequenceNumber;
use types::committee::EpochId;
use types::config::node_config::AuthorityStorePruningConfig;
use types::digests::TransactionDigest;
use types::object::ObjectID;

use authority::authority_store_pruner::{
    AuthorityStorePruner, EPOCH_DURATION_MS_FOR_TESTING, PrunerWatermarks,
};
use authority::authority_store_tables::{AuthorityPerpetualTables, AuthorityPrunerTables};
use authority::checkpoints::CheckpointStore;

pub use db_dump::DumpOptions;

pub mod db_dump;
pub mod index_search;

#[derive(Parser)]
pub enum DbToolCommand {
    /// List all column families in the database
    ListTables,

    /// Dump the contents of a table
    Dump(DumpOptions),

    /// Show a summary of a table
    TableSummary(DumpOptions),

    /// List DB metadata (live files) for a store
    ListDBMetadata(DumpOptions),

    /// Print the last consensus index
    PrintLastConsensusIndex {
        /// The epoch to inspect
        #[arg(long)]
        epoch: EpochId,
    },

    /// Print a transaction and its effects
    PrintTransaction {
        /// The transaction digest (hex string)
        #[arg(long)]
        digest: String,
    },

    /// Print an object by its ID
    PrintObject {
        /// The object ID (hex string)
        #[arg(long)]
        id: String,
        /// Optional version to query
        #[arg(long)]
        version: Option<u64>,
        /// Show all versions of the object (including tombstones)
        #[arg(long)]
        all_versions: bool,
    },

    /// Print a checkpoint summary by sequence number
    PrintCheckpoint {
        /// The checkpoint sequence number
        #[arg(long)]
        sequence_number: CheckpointSequenceNumber,
    },

    /// Print checkpoint contents by sequence number
    PrintCheckpointContent {
        /// The checkpoint sequence number
        #[arg(long)]
        sequence_number: CheckpointSequenceNumber,
    },

    /// Reset the database for execution since genesis (destructive!)
    ResetDB,

    /// Rewind checkpoint execution to a specific point
    RewindCheckpointExecution {
        /// The epoch to rewind to
        #[arg(long)]
        epoch: EpochId,
        /// The checkpoint sequence number to rewind to
        #[arg(long)]
        checkpoint_sequence_number: CheckpointSequenceNumber,
    },

    /// Compact the objects table
    Compact,

    /// Prune old object versions
    PruneObjects,

    /// Prune old checkpoints
    PruneCheckpoints,

    /// Set the highest executed checkpoint watermark
    SetCheckpointWatermark {
        /// The checkpoint sequence number to set
        #[arg(long)]
        sequence_number: CheckpointSequenceNumber,
    },

    /// Search the owner index by address
    IndexSearchOwner(index_search::OwnerSearchOptions),

    /// Search the balance index by address
    IndexSearchBalance(index_search::BalanceSearchOptions),

    /// Search the target index
    IndexSearchTarget(index_search::TargetSearchOptions),
}

pub fn print_db_all_tables(db_path: PathBuf) -> anyhow::Result<()> {
    db_dump::list_tables(db_path)
}

pub async fn execute_db_tool_command(db_path: PathBuf, cmd: DbToolCommand) -> anyhow::Result<()> {
    match cmd {
        DbToolCommand::ListTables => {
            db_dump::list_tables(db_path)?;
        }

        DbToolCommand::Dump(opts) => {
            db_dump::dump_table(
                opts.store,
                opts.epoch,
                db_path,
                &opts.table_name,
                opts.page_size,
                opts.page_number,
            )?;
        }

        DbToolCommand::TableSummary(opts) => {
            db_dump::table_summary(opts.store, opts.epoch, db_path, &opts.table_name)?;
        }

        DbToolCommand::ListDBMetadata(opts) => {
            db_dump::print_table_metadata(opts.store, opts.epoch, db_path, &opts.table_name)?;
        }

        DbToolCommand::PrintLastConsensusIndex { epoch } => {
            print_last_consensus_index(&db_path, epoch)?;
        }

        DbToolCommand::PrintTransaction { digest } => {
            print_transaction(&db_path, &digest)?;
        }

        DbToolCommand::PrintObject { id, version, all_versions } => {
            print_object(&db_path, &id, version, all_versions)?;
        }

        DbToolCommand::PrintCheckpoint { sequence_number } => {
            print_checkpoint(&db_path, sequence_number)?;
        }

        DbToolCommand::PrintCheckpointContent { sequence_number } => {
            print_checkpoint_content(&db_path, sequence_number)?;
        }

        DbToolCommand::ResetDB => {
            reset_db(&db_path).await?;
        }

        DbToolCommand::RewindCheckpointExecution { epoch, checkpoint_sequence_number } => {
            rewind_checkpoint_execution(&db_path, epoch, checkpoint_sequence_number)?;
        }

        DbToolCommand::Compact => {
            compact(&db_path)?;
        }

        DbToolCommand::PruneObjects => {
            prune_objects(&db_path).await?;
        }

        DbToolCommand::PruneCheckpoints => {
            prune_checkpoints(&db_path).await?;
        }

        DbToolCommand::SetCheckpointWatermark { sequence_number } => {
            set_checkpoint_watermark(&db_path, sequence_number)?;
        }

        DbToolCommand::IndexSearchOwner(opts) => {
            index_search::search_owner_index(db_path, opts)?;
        }

        DbToolCommand::IndexSearchBalance(opts) => {
            index_search::search_balance_index(db_path, opts)?;
        }

        DbToolCommand::IndexSearchTarget(opts) => {
            index_search::search_target_index(db_path, opts)?;
        }
    }
    Ok(())
}

fn print_last_consensus_index(db_path: &PathBuf, epoch: EpochId) -> anyhow::Result<()> {
    let epoch_tables =
        authority::authority_per_epoch_store::AuthorityEpochTables::open(epoch, db_path, None);
    match epoch_tables.get_last_consensus_stats()? {
        Some(stats) => {
            println!("Last consensus index: {:?}", stats.index);
            println!("Stats: {:?}", stats.stats);
        }
        None => match epoch_tables.get_last_consensus_index()? {
            Some(index) => println!("Last consensus index: {:?}", index),
            None => println!("No consensus index found for epoch {}", epoch),
        },
    }
    Ok(())
}

fn hex_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    let trimmed = s.trim_start_matches("0x");
    if trimmed.len() % 2 != 0 {
        return Err(anyhow!("Hex string must have even length"));
    }
    (0..trimmed.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&trimmed[i..i + 2], 16).map_err(|e| anyhow!("Invalid hex: {}", e))
        })
        .collect()
}

fn parse_digest(digest_str: &str) -> anyhow::Result<TransactionDigest> {
    let bytes = hex_decode(digest_str)?;
    let arr: [u8; 32] = bytes.try_into().map_err(|v: Vec<u8>| {
        anyhow!("Invalid digest length: expected 32 bytes, got {}", v.len())
    })?;
    Ok(TransactionDigest::from(arr))
}

fn parse_object_id(id_str: &str) -> anyhow::Result<ObjectID> {
    let bytes = hex_decode(id_str)?;
    ObjectID::try_from(bytes.as_slice()).map_err(|e| anyhow!("Invalid object ID: {}", e))
}

fn print_transaction(db_path: &PathBuf, digest_str: &str) -> anyhow::Result<()> {
    let perpetual_db = AuthorityPerpetualTables::open(db_path, None);
    let digest = parse_digest(digest_str)?;

    // Print the transaction
    match perpetual_db.get_transaction(&digest)? {
        Some(tx) => {
            println!("Transaction: {:?}", tx);
        }
        None => {
            println!("Transaction not found: {}", digest);
        }
    }

    // Print effects
    match perpetual_db.get_effects(&digest)? {
        Some(effects) => {
            println!("Effects: {:?}", effects);
        }
        None => {
            println!("Effects not found for: {}", digest);
        }
    }

    Ok(())
}

fn print_object(
    db_path: &PathBuf,
    id_str: &str,
    version: Option<u64>,
    all_versions: bool,
) -> anyhow::Result<()> {
    let perpetual_db = AuthorityPerpetualTables::open(db_path, None);
    let object_id = parse_object_id(id_str)?;

    if all_versions {
        // Scan all versions of this object using the readonly handle
        let ro = AuthorityPerpetualTables::open_readonly(db_path);
        let mut count = 0u64;
        for entry in ro.objects.safe_iter_with_bounds(
            Some(types::storage::ObjectKey::min_for_id(&object_id)),
            Some(types::storage::ObjectKey::max_for_id(&object_id)),
        ) {
            let (key, value) = entry?;
            if key.0 != object_id {
                break;
            }
            println!("  Version {} => {:?}", key.1.value(), value);
            count += 1;
        }
        if count == 0 {
            println!("No versions found for {}", object_id);
        } else {
            println!("\nTotal: {} versions", count);
        }
    } else if let Some(v) = version {
        let version = types::object::Version::from_u64(v);
        match perpetual_db.get_object_by_key_fallible(&object_id, version)? {
            Some(obj) => println!("Object: {:?}", obj),
            None => println!("Object not found at version {}", v),
        }
    } else {
        match perpetual_db.find_object_lt_or_eq_version(object_id, types::object::Version::MAX)? {
            Some(obj) => println!("Object: {:?}", obj),
            None => println!("Object not found: {}", object_id),
        }
    }

    Ok(())
}

fn print_checkpoint(
    db_path: &PathBuf,
    sequence_number: CheckpointSequenceNumber,
) -> anyhow::Result<()> {
    let checkpoint_db =
        CheckpointStore::new(&db_path.join("checkpoints"), Arc::new(PrunerWatermarks::default()));
    match checkpoint_db.get_checkpoint_by_sequence_number(sequence_number)? {
        Some(checkpoint) => {
            println!("Checkpoint: {:?}", checkpoint);
        }
        None => {
            println!("Checkpoint not found: {}", sequence_number);
        }
    }
    Ok(())
}

fn print_checkpoint_content(
    db_path: &PathBuf,
    sequence_number: CheckpointSequenceNumber,
) -> anyhow::Result<()> {
    let checkpoint_db =
        CheckpointStore::new(&db_path.join("checkpoints"), Arc::new(PrunerWatermarks::default()));

    let checkpoint = checkpoint_db
        .get_checkpoint_by_sequence_number(sequence_number)?
        .ok_or_else(|| anyhow!("Checkpoint not found: {}", sequence_number))?;

    let contents_digest = checkpoint.content_digest;
    match checkpoint_db.get_checkpoint_contents(&contents_digest)? {
        Some(contents) => {
            println!("Checkpoint Contents: {:?}", contents);
        }
        None => {
            println!("Checkpoint contents not found for checkpoint {}", sequence_number);
        }
    }

    Ok(())
}

async fn reset_db(db_path: &PathBuf) -> anyhow::Result<()> {
    info!("Resetting database at {:?}", db_path);

    let perpetual_path = AuthorityPerpetualTables::path(db_path);
    store::rocks::safe_drop_db(perpetual_path.clone(), Duration::from_secs(30)).await?;
    info!("Dropped perpetual tables");

    let checkpoint_db =
        CheckpointStore::new(&db_path.join("checkpoints"), Arc::new(PrunerWatermarks::default()));
    checkpoint_db.reset_db_for_execution_since_genesis()?;
    info!("Reset checkpoint execution to genesis");

    println!("Database reset complete.");
    Ok(())
}

fn rewind_checkpoint_execution(
    db_path: &PathBuf,
    epoch: EpochId,
    checkpoint_sequence_number: CheckpointSequenceNumber,
) -> anyhow::Result<()> {
    let checkpoint_db =
        CheckpointStore::new(&db_path.join("checkpoints"), Arc::new(PrunerWatermarks::default()));

    let checkpoint =
        checkpoint_db
            .get_checkpoint_by_sequence_number(checkpoint_sequence_number)?
            .ok_or_else(|| anyhow!("Checkpoint not found: {}", checkpoint_sequence_number))?;

    checkpoint_db.set_highest_executed_checkpoint_subtle(&checkpoint)?;
    info!(
        "Rewound checkpoint execution to epoch {} checkpoint {}",
        epoch, checkpoint_sequence_number
    );
    println!("Successfully rewound to epoch {} checkpoint {}", epoch, checkpoint_sequence_number);

    Ok(())
}

fn compact(db_path: &PathBuf) -> anyhow::Result<()> {
    info!("Compacting perpetual tables at {:?}", db_path);
    let perpetual_db = Arc::new(AuthorityPerpetualTables::open(db_path, None));
    AuthorityStorePruner::compact(&perpetual_db)?;
    println!("Compaction complete.");
    Ok(())
}

async fn prune_objects(db_path: &PathBuf) -> anyhow::Result<()> {
    info!("Pruning objects at {:?}", db_path);
    let perpetual_db = Arc::new(AuthorityPerpetualTables::open(db_path, None));
    let checkpoint_db =
        CheckpointStore::new(&db_path.join("checkpoints"), Arc::new(PrunerWatermarks::default()));
    let pruner_db = Arc::new(AuthorityPrunerTables::open(db_path));
    let config = AuthorityStorePruningConfig::default();

    AuthorityStorePruner::prune_objects_for_eligible_epochs(
        &perpetual_db,
        &checkpoint_db,
        None, // rpc_index
        Some(&pruner_db),
        config,
        EPOCH_DURATION_MS_FOR_TESTING,
    )
    .await?;

    println!("Object pruning complete.");
    Ok(())
}

async fn prune_checkpoints(db_path: &PathBuf) -> anyhow::Result<()> {
    info!("Pruning checkpoints at {:?}", db_path);
    let perpetual_db = Arc::new(AuthorityPerpetualTables::open(db_path, None));
    let pruner_watermarks = Arc::new(PrunerWatermarks::default());
    let checkpoint_db =
        CheckpointStore::new(&db_path.join("checkpoints"), pruner_watermarks.clone());
    let pruner_db = Arc::new(AuthorityPrunerTables::open(db_path));
    let config = AuthorityStorePruningConfig::default();

    AuthorityStorePruner::prune_checkpoints_for_eligible_epochs(
        &perpetual_db,
        &checkpoint_db,
        None, // rpc_index
        Some(&pruner_db),
        config,
        EPOCH_DURATION_MS_FOR_TESTING,
        &pruner_watermarks,
    )
    .await?;

    println!("Checkpoint pruning complete.");
    Ok(())
}

fn set_checkpoint_watermark(
    db_path: &PathBuf,
    sequence_number: CheckpointSequenceNumber,
) -> anyhow::Result<()> {
    let checkpoint_db =
        CheckpointStore::new(&db_path.join("checkpoints"), Arc::new(PrunerWatermarks::default()));

    let checkpoint = checkpoint_db
        .get_checkpoint_by_sequence_number(sequence_number)?
        .ok_or_else(|| anyhow!("Checkpoint not found: {}", sequence_number))?;

    checkpoint_db.set_highest_executed_checkpoint_subtle(&checkpoint)?;
    println!("Set highest executed checkpoint watermark to {}", sequence_number);

    Ok(())
}
