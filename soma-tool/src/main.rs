// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::Parser;

use soma_tool::db_tool;

bin_version::bin_version!();

#[derive(Parser)]
#[command(
    name = "soma-tool",
    about = "Soma DB inspection and maintenance tool",
    version = VERSION
)]
struct Args {
    /// Path to the authority database
    #[arg(long = "db-path")]
    db_path: String,
    #[command(subcommand)]
    cmd: Option<db_tool::DbToolCommand>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let path = PathBuf::from(args.db_path);

    match args.cmd {
        Some(cmd) => db_tool::execute_db_tool_command(path, cmd).await,
        None => db_tool::print_db_all_tables(path),
    }
}
