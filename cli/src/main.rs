use clap::Parser;
use cli::{Cli, Commands};
use shard::Encoder;
use tokio::signal;

mod cli;

#[tokio::main]
async fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Start {
            python_project_root,
            entry_point,
            port,
        } => {
            // figure out the necessary shit
            tokio::select! {
                _ = signal::ctrl_c() => {
                    println!("Shutting down server");
                }
                // _ = Encoder::start(port) => {
                //     // Handle server completion if needed
                // }
            }
        }
    }
}
