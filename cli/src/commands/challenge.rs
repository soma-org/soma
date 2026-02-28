// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::fmt::{self, Display, Formatter};

use anyhow::{Result, anyhow, bail};
use clap::*;
use colored::Colorize;
use sdk::wallet_context::WalletContext;
use serde::Serialize;
use tabled::builder::Builder as TableBuilder;
use tabled::settings::style::HorizontalLine;
use tabled::settings::{Panel as TablePanel, Style as TableStyle};
use types::base::SomaAddress;
use types::challenge::{ChallengeId, ChallengeStatus, ChallengeV1};
use types::object::ObjectID;
use types::target::TargetId;
use types::transaction::{InitiateChallengeArgs, TransactionKind};

use types::system_state::SystemStateTrait as _;

use crate::client_commands::TxProcessingArgs;
use crate::response::TransactionResponse;

#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum ChallengeCommand {
    /// Submit a fraud challenge against a filled target
    ///
    /// Challenges a submission for a filled target. A bond coin is auto-selected.
    /// If the challenge succeeds (fraud detected), the challenger receives the
    /// submitter's bond. If it fails, the challenger loses their bond to validators.
    ///
    /// Requirements:
    /// - Target must be filled (have a winning submission)
    /// - Challenge must be submitted during the fill epoch (before epoch boundary)
    /// - Sender must have a coin covering challenger_bond_per_byte * data_size
    #[clap(
        name = "submit",
        after_help = "\
EXAMPLES:
    soma challenge submit --target-id 0xTARGET_ID"
    )]
    Submit {
        /// Target ID to challenge
        #[clap(long)]
        target_id: ObjectID,
        #[clap(flatten)]
        tx_args: TxProcessingArgs,
    },

    /// Show challenge information
    ///
    /// Displays detailed information about a specific challenge.
    #[clap(name = "info")]
    Info {
        /// Challenge ID to query
        challenge_id: ObjectID,
    },

    /// List challenges (not yet implemented)
    ///
    /// Challenge indexing is not yet available. Use `soma challenge info <id>`
    /// to query a specific challenge by its object ID.
    #[clap(name = "list", hide = true)]
    List,
}

// =============================================================================
// Execution
// =============================================================================

impl ChallengeCommand {
    pub async fn execute(self, context: &mut WalletContext) -> Result<ChallengeCommandResponse> {
        match self {
            ChallengeCommand::Submit {
                target_id,
                tx_args,
            } => {
                let sender = context.active_address()?;

                // Auto-fetch bond coin
                let bond_coin_ref =
                    super::parse_helpers::auto_fetch_bond_coin(context, sender).await?;

                // Pre-fetch epoch timing (non-fatal, before transaction)
                let resolution_hint = match context.get_client().await {
                    Ok(client) => client
                        .get_latest_system_state()
                        .await
                        .ok()
                        .and_then(|s| {
                            crate::response::format_next_epoch_hint(
                                s.epoch_start_timestamp_ms(),
                                s.epoch_duration_ms(),
                            )
                        })
                        .map(|h| {
                            format!("Challenge resolves at epoch boundary ({h})")
                        }),
                    Err(_) => None,
                };

                // ChallengeId is derived from tx_digest during execution, not client-provided
                let kind = TransactionKind::InitiateChallenge(InitiateChallengeArgs {
                    target_id,
                    bond_coin: bond_coin_ref,
                });

                execute_tx(context, sender, kind, tx_args, resolution_hint).await
            }

            ChallengeCommand::Info { challenge_id } => {
                let client = context.get_client().await?;

                // Query the challenge object directly
                let object = client
                    .get_object(challenge_id)
                    .await
                    .map_err(|e| anyhow!("Failed to get challenge object: {}", e.message()))?;

                // Deserialize the challenge from object contents
                let challenge: ChallengeV1 = bcs::from_bytes(object.data.contents())
                    .map_err(|e| anyhow!("Failed to deserialize challenge: {}", e))?;

                Ok(ChallengeCommandResponse::Info(ChallengeInfoOutput {
                    challenge_id,
                    challenge,
                }))
            }

            ChallengeCommand::List => {
                Ok(ChallengeCommandResponse::List(ChallengeListOutput {
                    challenges: vec![],
                    message: Some(
                        "Challenge indexing is not yet implemented. Use 'soma challenge info <challenge_id>' \
                        to query a specific challenge by ID.".to_string()
                    ),
                }))
            }
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Execute a challenge transaction, delegating to the shared client_commands helper.
/// The ChallengeId is derived from the transaction effects (created objects).
async fn execute_tx(
    context: &mut WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
    tx_args: TxProcessingArgs,
    resolution_hint: Option<String>,
) -> Result<ChallengeCommandResponse> {
    let result =
        crate::client_commands::execute_or_serialize(context, sender, kind, None, tx_args).await?;

    // Convert ClientCommandResponse to ChallengeCommandResponse
    match result {
        crate::response::ClientCommandResponse::Transaction(tx) => {
            // Extract challenge_id from created objects in effects
            // The Challenge is the shared object created by InitiateChallenge
            let challenge_id = tx
                .created
                .first()
                .map(|obj| obj.object_id)
                .ok_or_else(|| anyhow!("No challenge object created in transaction effects"))?;
            Ok(ChallengeCommandResponse::Initiated(ChallengeInitiatedOutput {
                challenge_id,
                transaction: tx,
                resolution_hint,
            }))
        }
        crate::response::ClientCommandResponse::SerializedUnsignedTransaction(s) => {
            // When serializing, we don't know the challenge_id yet (it's derived from tx_digest)
            Ok(ChallengeCommandResponse::SerializedTransaction { serialized_transaction: s })
        }
        crate::response::ClientCommandResponse::SerializedSignedTransaction(s) => {
            Ok(ChallengeCommandResponse::SerializedTransaction { serialized_transaction: s })
        }
        crate::response::ClientCommandResponse::TransactionDigest(d) => {
            // When only digest is returned, we can derive the challenge_id
            // ChallengeId = ObjectID::derive_id(tx_digest, 0) for the first created object
            let challenge_id = ObjectID::derive_id(d, 0);
            Ok(ChallengeCommandResponse::TransactionDigest { challenge_id, digest: d })
        }
        crate::response::ClientCommandResponse::Simulation(sim) => {
            Ok(ChallengeCommandResponse::Simulation(sim))
        }
        _ => bail!("Unexpected response type from transaction execution"),
    }
}

// =============================================================================
// Response types
// =============================================================================

#[derive(Debug, Serialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum ChallengeCommandResponse {
    Initiated(ChallengeInitiatedOutput),
    Info(ChallengeInfoOutput),
    List(ChallengeListOutput),
    /// Serialized transaction (challenge_id not known until execution)
    SerializedTransaction {
        serialized_transaction: String,
    },
    TransactionDigest {
        challenge_id: ChallengeId,
        digest: types::digests::TransactionDigest,
    },
    Simulation(crate::response::SimulationResponse),
}

#[derive(Debug, Serialize)]
pub struct ChallengeInitiatedOutput {
    pub challenge_id: ChallengeId,
    pub transaction: TransactionResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution_hint: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChallengeInfoOutput {
    pub challenge_id: ChallengeId,
    pub challenge: ChallengeV1,
}

#[derive(Debug, Serialize)]
pub struct ChallengeListOutput {
    pub challenges: Vec<ChallengeSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChallengeSummary {
    pub challenge_id: ChallengeId,
    pub target_id: TargetId,
    pub challenger: SomaAddress,
    pub status: String,
    pub challenge_epoch: u64,
    pub challenger_bond: u64,
}

// =============================================================================
// Display implementations
// =============================================================================

impl Display for ChallengeCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ChallengeCommandResponse::Initiated(output) => write!(f, "{}", output),
            ChallengeCommandResponse::Info(info) => write!(f, "{}", info),
            ChallengeCommandResponse::List(list) => write!(f, "{}", list),
            ChallengeCommandResponse::SerializedTransaction { serialized_transaction } => {
                writeln!(f, "{}", "Serialized Transaction".cyan().bold())?;
                writeln!(f)?;
                writeln!(f, "{}", serialized_transaction)?;
                writeln!(f)?;
                writeln!(
                    f,
                    "{}",
                    "Challenge ID will be derived from transaction digest after signing and execution.".yellow()
                )?;
                writeln!(f, "{}", "Use 'soma tx execute-signed' to submit after signing.".yellow())
            }
            ChallengeCommandResponse::TransactionDigest { challenge_id, digest } => {
                writeln!(f, "{}: {}", "Challenge ID".bold(), challenge_id)?;
                writeln!(f, "{}: {}", "Transaction Digest".bold(), digest)
            }
            ChallengeCommandResponse::Simulation(sim) => write!(f, "{}", sim),
        }
    }
}

impl Display for ChallengeInitiatedOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", "Challenge Initiated".green().bold())?;
        writeln!(f)?;
        writeln!(f, "{}: {}", "Challenge ID".bold(), self.challenge_id)?;
        if let Some(hint) = &self.resolution_hint {
            writeln!(f, "  {} {}", ">>".dimmed(), hint.yellow())?;
        }
        writeln!(f)?;
        write!(f, "{}", self.transaction)
    }
}

impl Display for ChallengeInfoOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let c = &self.challenge;
        let mut builder = TableBuilder::default();

        builder.push_record(["Challenge ID", &self.challenge_id.to_string()]);
        builder.push_record(["Target ID", &c.target_id.to_string()]);
        builder.push_record(["Challenger", &c.challenger.to_string()]);
        builder.push_record(["Status", &format_status(&c.status)]);
        builder.push_record(["Challenge Epoch", &c.challenge_epoch.to_string()]);
        builder.push_record([
            "Challenger Bond",
            &format!(
                "{} shannons ({})",
                c.challenger_bond,
                crate::response::format_soma(c.challenger_bond as u128),
            ),
        ]);
        builder.push_record(["Distance Threshold", &c.distance_threshold.to_string()]);
        builder.push_record(["Claimed Distance", &c.winning_distance_score.to_string()]);
        builder.push_record(["Winning Model", &c.winning_model_id.to_string()]);

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header("Challenge Information"));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);

        writeln!(f, "{}", table)?;

        // Show assigned models
        if !c.model_ids.is_empty() {
            writeln!(f)?;
            writeln!(f, "{}", "Target's Assigned Models:".cyan().bold())?;
            for (i, model_id) in c.model_ids.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, model_id)?;
            }
        }

        Ok(())
    }
}

impl Display for ChallengeListOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(msg) = &self.message {
            return writeln!(f, "{}", msg.yellow());
        }

        if self.challenges.is_empty() {
            return writeln!(f, "{}", "No challenges found.".yellow());
        }

        let mut builder = TableBuilder::default();
        builder.push_record(["Challenge ID", "Target ID", "Challenger", "Status", "Epoch", "Bond"]);

        for c in &self.challenges {
            builder.push_record([
                truncate_id(&c.challenge_id.to_string()),
                truncate_id(&c.target_id.to_string()),
                truncate_id(&c.challenger.to_string()),
                c.status.clone(),
                c.challenge_epoch.to_string(),
                crate::response::format_soma(c.challenger_bond as u128),
            ]);
        }

        let mut table = builder.build();
        table.with(TableStyle::rounded());
        table.with(TablePanel::header(format!("Challenges ({} total)", self.challenges.len())));
        table.with(HorizontalLine::new(1, TableStyle::modern().get_horizontal()));
        table.with(HorizontalLine::new(2, TableStyle::modern().get_horizontal()));
        table.with(tabled::settings::style::BorderSpanCorrection);
        writeln!(f, "{}", table)
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn format_status(status: &ChallengeStatus) -> String {
    match status {
        ChallengeStatus::Pending => "Pending".yellow().to_string(),
        ChallengeStatus::Resolved { challenger_lost } => {
            if *challenger_lost {
                format!("{} (Challenger Lost)", "Resolved".red())
            } else {
                format!("{} (Challenger Won)", "Resolved".green())
            }
        }
    }
}

use crate::response::truncate_id;

impl ChallengeCommandResponse {
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
