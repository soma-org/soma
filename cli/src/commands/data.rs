// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! CLI commands for downloading submission data from the validator network.
//!
//! This module provides the `soma target download` command for fetching data associated
//! with filled targets from the validator proxy network.

use anyhow::{Result, anyhow, bail};
use clap::*;
use colored::Colorize;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

use sdk::proxy_client::ProxyClient;
use sdk::wallet_context::WalletContext;
use types::object::ObjectID;
use types::target::TargetId;

/// Download submission data for a filled target
///
/// Fetches the winning submission's data from the validator proxy network.
/// The target must be filled (have a winning submission) for this to work.
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub struct DataCommand {
    /// Target ID to download submission data for
    pub target_id: ObjectID,
    /// Output file path (defaults to ./<target_id>.data)
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

// =============================================================================
// Execution
// =============================================================================

impl DataCommand {
    pub async fn execute(self, context: &mut WalletContext) -> Result<DataCommandResponse> {
        let client = context.get_client().await?;

        // Get system state to create proxy client
        let system_state = client
            .get_latest_system_state()
            .await
            .map_err(|e| anyhow!("Failed to get system state: {}", e))?;

        // Create proxy client from system state
        let proxy_client = ProxyClient::from_system_state(&system_state)
            .map_err(|e| anyhow!("Failed to create proxy client: {}", e))?;

        if proxy_client.validator_count() == 0 {
            bail!("No validators with proxy addresses available");
        }

        // Determine output path
        let output_path =
            self.output.unwrap_or_else(|| PathBuf::from(format!("{}.data", self.target_id)));

        // Download submission data
        eprintln!(
            "Downloading submission data for target {} from {} validators...",
            self.target_id,
            proxy_client.validator_count()
        );

        let data = proxy_client
            .fetch_submission_data(&self.target_id)
            .await
            .map_err(|e| anyhow!("Failed to download submission data: {}", e))?;

        // Write to file
        let mut file = tokio::fs::File::create(&output_path)
            .await
            .map_err(|e| anyhow!("Failed to create output file: {}", e))?;
        file.write_all(&data).await.map_err(|e| anyhow!("Failed to write to file: {}", e))?;

        Ok(DataCommandResponse::Downloaded(DataDownloadOutput {
            target_id: self.target_id,
            output_path,
            size_bytes: data.len(),
        }))
    }
}

// =============================================================================
// Response types
// =============================================================================

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum DataCommandResponse {
    Downloaded(DataDownloadOutput),
}

#[derive(Debug, Serialize)]
pub struct DataDownloadOutput {
    pub target_id: TargetId,
    pub output_path: PathBuf,
    pub size_bytes: usize,
}

// =============================================================================
// Display implementations
// =============================================================================

impl Display for DataCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DataCommandResponse::Downloaded(dl) => write!(f, "{}", dl),
        }
    }
}

impl Display for DataDownloadOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", "Data Downloaded Successfully".green().bold())?;
        writeln!(f)?;
        writeln!(f, "{}: {}", "Target ID".bold(), self.target_id)?;
        writeln!(f, "{}: {}", "Output".bold(), self.output_path.display())?;
        writeln!(f, "{}: {} bytes", "Size".bold(), self.size_bytes)
    }
}

impl DataCommandResponse {
    pub fn print(&self, json: bool) {
        if json {
            match serde_json::to_string_pretty(self) {
                Ok(s) => println!("{}", s),
                Err(e) => eprintln!("Failed to serialize response: {}", e),
            }
        } else {
            print!("{}", self);
        }
    }
}
