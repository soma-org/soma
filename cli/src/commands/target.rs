// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow};
use clap::*;
use colored::Colorize;
use rpc::proto::soma::ListTargetsRequest;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use tabled::{
    builder::Builder as TableBuilder,
    settings::{Panel as TablePanel, Style as TableStyle, style::HorizontalLine},
};

use sdk::wallet_context::WalletContext;
use types::{
    object::ObjectID,
    target::{TargetId, TargetStatus, TargetV1},
};

use super::claim::ClaimCommand;
use super::data::DataCommand;
use super::submit::SubmitCommand;

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum TargetCommand {
    /// List targets
    ///
    /// Shows targets with optional filtering by status and epoch.
    #[clap(name = "list")]
    List {
        /// Filter by status: "open", "filled", or "claimed"
        #[clap(long, short = 's')]
        status: Option<String>,
        /// Filter by generation epoch
        #[clap(long, short = 'e')]
        epoch: Option<u64>,
        /// Maximum number of targets to return (default: 50, max: 1000)
        #[clap(long, default_value = "50")]
        limit: u32,
    },

    /// Show target information
    ///
    /// Displays detailed information about a specific target.
    #[clap(name = "info")]
    Info {
        /// Target ID to query
        target_id: ObjectID,
    },

    /// Submit data to fill a target
    #[clap(
        name = "submit",
        after_help = "\
EXAMPLES:
    soma target submit --target-id 0xTARGET... --data-commitment 0xHEX... \\
        --data-url https://... --data-checksum 0xHEX... --data-size 1024 \\
        --model-id 0xMODEL... --embedding 0.1,0.2,0.3 \\
        --distance-score 0.5 --bond-coin 0xCOIN..."
    )]
    Submit {
        #[clap(flatten)]
        cmd: SubmitCommand,
    },

    /// Claim rewards from a filled target
    ///
    /// Claims the reward pool from a target that was successfully filled.
    /// The challenge window (one full epoch after the target was filled) must have closed.
    #[clap(
        name = "claim",
        after_help = "\
EXAMPLES:
    soma target claim 0xTARGET_ID"
    )]
    Claim {
        #[clap(flatten)]
        cmd: ClaimCommand,
    },

    /// Download submission data for a filled target
    ///
    /// Fetches the winning submission's data via the validator proxy network.
    #[clap(
        name = "download",
        after_help = "\
EXAMPLES:
    soma target download 0xTARGET_ID
    soma target download 0xTARGET_ID --output ./my-data.bin"
    )]
    Download {
        #[clap(flatten)]
        cmd: DataCommand,
    },
}

// =============================================================================
// Execution
// =============================================================================

impl TargetCommand {
    pub async fn execute(self, context: &mut WalletContext) -> Result<TargetCommandResponse> {
        match self {
            TargetCommand::List { status, epoch, limit } => {
                let client = context.get_client().await?;

                let mut request = ListTargetsRequest::default();
                request.status_filter = status;
                request.epoch_filter = epoch;
                request.page_size = Some(limit);

                let response = client
                    .list_targets(request)
                    .await
                    .map_err(|e| anyhow!("Failed to list targets: {}", e))?;

                let targets: Vec<TargetSummary> = response
                    .targets
                    .into_iter()
                    .filter_map(|t| {
                        let target_id =
                            t.id.as_ref().and_then(|s| ObjectID::from_hex_literal(s).ok())?;
                        Some(TargetSummary {
                            target_id,
                            status: t.status.unwrap_or_else(|| "unknown".to_string()),
                            generation_epoch: t.generation_epoch.unwrap_or(0),
                            model_count: t.model_ids.len(),
                            reward_pool: t.reward_pool.unwrap_or(0),
                            distance_threshold: t.distance_threshold.unwrap_or(0.0),
                        })
                    })
                    .collect();

                Ok(TargetCommandResponse::List(TargetListOutput { targets, message: None }))
            }

            TargetCommand::Info { target_id } => {
                let client = context.get_client().await?;

                // Query the target object directly
                let object = client
                    .get_object(target_id)
                    .await
                    .map_err(|e| anyhow!("Failed to get target object: {}", e))?;

                // Deserialize the target from object contents
                let target: TargetV1 = bcs::from_bytes(object.data.contents())
                    .map_err(|e| anyhow!("Failed to deserialize target: {}", e))?;

                Ok(TargetCommandResponse::Info(TargetInfoOutput { target_id, target }))
            }

            TargetCommand::Submit { cmd } => {
                let result = cmd.execute(context).await?;
                Ok(TargetCommandResponse::Submit(result))
            }

            TargetCommand::Claim { cmd } => {
                let result = cmd.execute(context).await?;
                Ok(TargetCommandResponse::Claim(result))
            }

            TargetCommand::Download { cmd } => {
                let result = cmd.execute(context).await?;
                Ok(TargetCommandResponse::Download(result))
            }
        }
    }
}

// =============================================================================
// Response types
// =============================================================================

#[derive(Debug, Serialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum TargetCommandResponse {
    List(TargetListOutput),
    Info(TargetInfoOutput),
    Submit(super::submit::SubmitCommandResponse),
    Claim(super::claim::ClaimCommandResponse),
    Download(super::data::DataCommandResponse),
}

#[derive(Debug, Serialize)]
pub struct TargetListOutput {
    pub targets: Vec<TargetSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetSummary {
    pub target_id: TargetId,
    pub status: String,
    pub generation_epoch: u64,
    pub model_count: usize,
    pub reward_pool: u64,
    pub distance_threshold: f32,
}

#[derive(Debug, Serialize)]
pub struct TargetInfoOutput {
    pub target_id: TargetId,
    pub target: TargetV1,
}

// =============================================================================
// Display implementations
// =============================================================================

impl Display for TargetCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            TargetCommandResponse::List(list) => write!(f, "{}", list),
            TargetCommandResponse::Info(info) => write!(f, "{}", info),
            TargetCommandResponse::Submit(resp) => write!(f, "{}", resp),
            TargetCommandResponse::Claim(resp) => write!(f, "{}", resp),
            TargetCommandResponse::Download(resp) => write!(f, "{}", resp),
        }
    }
}

impl Display for TargetListOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(msg) = &self.message {
            return writeln!(f, "{}", msg.yellow());
        }

        if self.targets.is_empty() {
            return writeln!(f, "{}", "No targets found.".yellow());
        }

        let mut builder = TableBuilder::default();
        builder.push_record([
            "Target ID",
            "Status",
            "Epoch",
            "Models",
            "Reward",
            "Distance Thresh",
        ]);

        for t in &self.targets {
            builder.push_record([
                truncate_id(&t.target_id.to_string()),
                t.status.clone(),
                t.generation_epoch.to_string(),
                t.model_count.to_string(),
                crate::response::format_soma(t.reward_pool as u128),
                t.distance_threshold.to_string(),
            ]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header(format!("Targets ({} total)", self.targets.len())));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)
    }
}

impl Display for TargetInfoOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let t = &self.target;
        let mut builder = TableBuilder::default();

        builder.push_record(["Target ID", &self.target_id.to_string()]);
        builder.push_record(["Status", &format_status(&t.status)]);
        builder.push_record(["Generation Epoch", &t.generation_epoch.to_string()]);
        builder.push_record(["Embedding Dimension", &t.embedding.len().to_string()]);
        builder.push_record(["Model Count", &t.model_ids.len().to_string()]);
        builder.push_record(["Distance Threshold", &t.distance_threshold.to_string()]);
        builder.push_record([
            "Reward Pool",
            &format!(
                "{} shannons ({})",
                t.reward_pool,
                crate::response::format_soma(t.reward_pool as u128),
            ),
        ]);

        if let Some(submitter) = &t.submitter {
            builder.push_record(["Submitter", &submitter.to_string()]);
        }
        if let Some(model_id) = &t.winning_model_id {
            builder.push_record(["Winning Model", &model_id.to_string()]);
        }
        if let Some(owner) = &t.winning_model_owner {
            builder.push_record(["Model Owner", &owner.to_string()]);
        }
        if t.bond_amount > 0 {
            builder.push_record([
                "Bond Amount",
                &format!(
                    "{} shannons ({})",
                    t.bond_amount,
                    crate::response::format_soma(t.bond_amount as u128),
                ),
            ]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Target Information"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);

        writeln!(f, "{}", table)?;

        // Show assigned models
        if !t.model_ids.is_empty() {
            writeln!(f)?;
            writeln!(f, "{}", "Assigned Models:".cyan().bold())?;
            for (i, model_id) in t.model_ids.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, model_id)?;
            }
        }

        Ok(())
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn format_status(status: &TargetStatus) -> String {
    match status {
        TargetStatus::Open => "Open".green().to_string(),
        TargetStatus::Filled { fill_epoch } => {
            format!("{} (epoch {})", "Filled".yellow(), fill_epoch)
        }
        TargetStatus::Claimed => "Claimed".green().to_string(),
    }
}

use crate::response::truncate_id;

impl TargetCommandResponse {
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
