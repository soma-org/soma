// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use clap::*;
use cli::soma_commands::SomaCommand;
use colored::Colorize;

use tracing::debug;

// Define the `GIT_REVISION` and `VERSION` consts
bin_version::bin_version!();

/// Format an anyhow error chain for user-friendly display.
///
/// - Shows the root cause prominently
/// - Strips the verbose `Debug` chain in favor of a clean message
/// - Adds contextual hints for common error patterns
fn format_error(err: &anyhow::Error) -> String {
    // If the root cause is a tonic::Status, extract just the message
    // to avoid leaking gRPC internals (details, metadata) to users.
    let message = if let Some(status) = err.root_cause().downcast_ref::<tonic::Status>() {
        status.message().to_string()
    } else {
        let root = err.root_cause().to_string();
        let display = err.to_string();
        if display == root {
            display
        } else {
            // Show the root cause for clarity when there's context wrapping
            format!("{}: {}", display, root)
        }
    };

    let mut output = format!("{} {}", "Error:".red().bold(), message);

    // Append hint for common error patterns
    if let Some(hint) = error_hint(&message) {
        output.push_str(&format!("\n\n  {} {}", "Hint:".yellow().bold(), hint));
    }

    output
}

/// Return a contextual hint for common error messages.
fn error_hint(msg: &str) -> Option<&'static str> {
    let msg_lower = msg.to_lowercase();

    if msg_lower.contains("commission rate cannot exceed") {
        return Some("Commission rates are in basis points. Use 500 for 5%.");
    }
    if msg_lower.contains("no soma config found") || msg_lower.contains("cannot open soma") {
        return Some("Run `soma start localnet` to launch a local network first.");
    }
    if msg_lower.contains("connection refused") || msg_lower.contains("status: unavailable") {
        return Some("Is the network running? Try `soma start localnet` to launch a local network.");
    }
    if msg_lower.contains("insufficient fund") || msg_lower.contains("insufficient gas") {
        return Some("Use `soma faucet` to request test tokens.");
    }
    if msg_lower.contains("force-regenesis") {
        return Some(
            "Use --force-regenesis for an ephemeral network, or remove ~/.soma/ to start fresh.",
        );
    }
    if msg_lower.contains("not found in active, pending, or inactive") {
        return Some("Check the model ID with `soma model list`.");
    }
    if msg_lower.contains("must be exactly 32 bytes") {
        return Some(
            "Hex values should be 64 hex characters (32 bytes). Include the 0x prefix or omit it.",
        );
    }
    None
}

#[derive(Parser)]
#[clap(
    name = env!("CARGO_BIN_NAME"),
    about = "A game of infinitely evolving machine intelligence",
    rename_all = "kebab-case",
    author,
    version = VERSION,
    propagate_version = false,
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
    let log_level = args.command.log_level();
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    debug!("Soma CLI version: {VERSION}");

    if let Err(err) = args.command.execute().await {
        eprintln!("{}", format_error(&err));
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_hints() {
        assert!(error_hint("Commission rate cannot exceed 10000").is_some());
        assert!(error_hint("Connection refused (os error 61)").is_some());
        assert!(error_hint("No soma config found in `/Users/foo`").is_some());
        assert!(error_hint("some random error").is_none());
    }

    #[test]
    fn test_format_error_strips_grpc_metadata() {
        let status = tonic::Status::unavailable("tcp connect error");
        let err: anyhow::Error = status.into();
        let formatted = format_error(&err);
        assert!(formatted.contains("tcp connect error"), "Should contain message: {formatted}");
        assert!(!formatted.contains("MetadataMap"), "Should not leak metadata: {formatted}");
        assert!(!formatted.contains("details: []"), "Should not leak details: {formatted}");
    }
}
