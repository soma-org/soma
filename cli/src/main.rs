// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

mod commands;
mod error;
mod key_identity;
mod keytool;

use clap::*;
use colored::Colorize;
use commands::SomaCommand;
use tracing::debug;

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
            // TODO: set up telemetry
        }
    };
    debug!("Soma CLI version: {VERSION}");
    exit_main!(args.command.execute().await);
}
