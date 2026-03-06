// Copyright (c) Mysten Labs, Inc.
// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use indexer_framework::IndexerArgs;
use indexer_framework::ingestion::{ClientArgs, IngestionConfig};
use indexer_framework::Indexer;
use indexer_framework::postgres::{Db, DbArgs};
use tracing::info;

/// Soma Postgres Indexer
///
/// Indexes Soma blockchain data into a Postgres database for efficient querying.
#[derive(Parser, Debug)]
#[command(name = "soma-indexer-alt", about = "Soma Postgres Indexer")]
struct Args {
    /// Database connection URL
    #[arg(long)]
    database_url: String,

    /// Indexer configuration (first/last checkpoint, pipeline selection)
    #[clap(flatten)]
    indexer_args: IndexerArgs,

    /// Ingestion client configuration (checkpoint source)
    #[clap(flatten)]
    client_args: ClientArgs,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let registry = prometheus::Registry::new();

    let db = Db::for_write(
        args.database_url
            .parse()
            .context("Failed to parse DATABASE_URL")?,
        DbArgs::default(),
    )
    .await
    .context("Failed to connect to database")?;

    // Run pending migrations
    db.run_migrations(Some(&indexer_alt_schema::MIGRATIONS))
        .await
        .context("Failed to run pending migrations")?;

    let mut indexer = Indexer::new(
        db,
        args.indexer_args,
        args.client_args,
        IngestionConfig::default(),
        None,
        &registry,
    )
    .await
    .context("Failed to create indexer")?;

    indexer_alt::setup_indexer(&mut indexer)
        .await
        .context("Failed to setup indexer pipelines")?;

    info!("Starting Soma indexer");

    let mut service = indexer.run().await.context("Failed to start indexer")?;
    service.join().await.context("Indexer service failed")
}
