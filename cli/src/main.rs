// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

mod commands;
mod inference;
mod keytool;
use clap::*;
use colored::Colorize;
use commands::SomaCommand;

use tracing::{debug, Level};

// Define the `GIT_REVISION` and `VERSION` consts
bin_version::bin_version!();

macro_rules! exit_main {
    ($result:expr) => {
        match $result {
            Ok(_) => (),
            Err(err) => {
                let err = format!("{:?}", err);
                println!("{}", err.bold().red());
                std::process::exit(1);
            }
        }
    };
}

#[derive(Parser)]
#[clap(
    name = env!("CARGO_BIN_NAME"),
    about = "A game of infinitely evolving machine intelligence",
    rename_all = "kebab-case",
    author,
    version = VERSION,
    propagate_version = true,
)]

struct Args {
    #[clap(subcommand)]
    command: SomaCommand,
}

#[tokio::main]
async fn main() {
    #[cfg(windows)]
    colored::control::set_virtual_terminal(true).unwrap();

    let args = Args::parse();
    let _guard = match args.command {
        _ => {
            tracing_subscriber::fmt()
                .with_max_level(Level::ERROR)
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .init();
        }
    };
    debug!("Soma CLI version: {VERSION}");
    exit_main!(args.command.execute().await);
}
