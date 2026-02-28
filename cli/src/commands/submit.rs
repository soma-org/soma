// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use std::fmt::{self, Display, Formatter};
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use clap::*;
use colored::Colorize;
use sdk::wallet_context::WalletContext;
use serde::Serialize;
use types::base::SomaAddress;
use types::checksum::Checksum;
use types::digests::DataCommitment;
use types::metadata::{Manifest, ManifestV1, Metadata, MetadataV1};
use types::object::ObjectID;
use types::submission::SubmissionManifest;
use types::tensor::SomaTensor;
use types::transaction::{SubmitDataArgs, TransactionKind};

use types::system_state::SystemStateTrait as _;

use crate::client_commands::TxProcessingArgs;
use crate::response::TransactionResponse;

/// Submit data to fill a target
///
/// Submits data that embeds within a target's radius. The submission
/// includes the data manifest, embedding vector, and reported scores.
/// A bond coin is auto-selected. Commitment, checksum, and size are
/// auto-computed from the data file.
#[derive(Parser)]
#[clap(
    rename_all = "kebab-case",
    after_help = "\
EXAMPLES:
    soma target submit --target-id 0xTARGET_ID \\
        --data-file ./data.bin \\
        --data-url https://storage.example.com/data.bin \\
        --model-id 0xMODEL_ID \\
        --embedding 0.1,0.2,0.3 \\
        --distance-score 0.5"
)]
pub struct SubmitCommand {
    /// Target ID to submit to
    #[clap(long)]
    pub target_id: ObjectID,
    /// Path to the data file (commitment, checksum, and size auto-computed)
    #[clap(long)]
    pub data_file: PathBuf,
    /// URL where the data is hosted
    #[clap(long)]
    pub data_url: String,
    /// Model ID to use for this submission (must be in target's model_ids)
    #[clap(long)]
    pub model_id: ObjectID,
    /// Embedding vector (comma-separated f32 values)
    #[clap(long)]
    pub embedding: String,
    /// Distance score (f32, must be <= target threshold)
    #[clap(long)]
    pub distance_score: f32,
    #[clap(flatten)]
    pub tx_args: TxProcessingArgs,
}

// =============================================================================
// Execution
// =============================================================================

impl SubmitCommand {
    pub async fn execute(self, context: &mut WalletContext) -> Result<SubmitCommandResponse> {
        let sender = context.active_address()?;

        // Auto-compute commitment, checksum, and size from data file
        let (commitment_bytes, checksum_hex, data_size) =
            super::parse_helpers::read_and_hash_file(&self.data_file)?;

        // Build data manifest
        let manifest = build_data_manifest(&self.data_url, &checksum_hex, data_size)?;

        // Parse embedding
        let embedding_vec = parse_embedding(&self.embedding)?;

        // Auto-fetch bond coin
        let bond_coin_ref = super::parse_helpers::auto_fetch_bond_coin(context, sender).await?;

        // Pre-fetch epoch timing (non-fatal, before transaction)
        let (next_epoch_hint, claim_epoch_hint) = match context.get_client().await {
            Ok(client) => match client.get_latest_system_state().await {
                Ok(s) => {
                    let start = s.epoch_start_timestamp_ms();
                    let dur = s.epoch_duration_ms();
                    let next = crate::response::format_next_epoch_hint(start, dur);
                    let claim =
                        crate::response::epoch_time_remaining_ms(start, dur).map(|remaining| {
                            let total = remaining.saturating_add(dur);
                            format!(
                                "Claiming available in {}",
                                crate::response::format_duration_approx(total)
                            )
                        });
                    (next, claim)
                }
                Err(_) => (None, None),
            },
            Err(_) => (None, None),
        };

        let kind = TransactionKind::SubmitData(SubmitDataArgs {
            target_id: self.target_id,
            data_commitment: DataCommitment::new(commitment_bytes),
            data_manifest: manifest,
            model_id: self.model_id,
            embedding: SomaTensor::new(embedding_vec.clone(), vec![embedding_vec.len()]),
            distance_score: SomaTensor::scalar(self.distance_score),
            bond_coin: bond_coin_ref,
        });

        let result = execute_tx(context, sender, kind, self.tx_args).await?;
        match result {
            SubmitCommandResponse::Transaction(tx) => Ok(SubmitCommandResponse::SubmitSuccess {
                transaction: tx,
                next_epoch_hint,
                claim_epoch_hint,
            }),
            other => Ok(other),
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

use super::parse_helpers::{parse_embedding, parse_hex_digest_32};

fn build_data_manifest(url: &str, checksum_hex: &str, size: usize) -> Result<SubmissionManifest> {
    let parsed_url: url::Url = url.parse().map_err(|e| anyhow!("Invalid URL: {}", e))?;
    let checksum_bytes = parse_hex_digest_32(checksum_hex, "data-checksum")?;

    let metadata = Metadata::V1(MetadataV1::new(Checksum(checksum_bytes), size));
    let manifest = Manifest::V1(ManifestV1::new(parsed_url, metadata));

    Ok(SubmissionManifest::new(manifest))
}

/// Execute a submit transaction, delegating to the shared client_commands helper.
async fn execute_tx(
    context: &mut WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
    tx_args: TxProcessingArgs,
) -> Result<SubmitCommandResponse> {
    let result =
        crate::client_commands::execute_or_serialize(context, sender, kind, None, tx_args).await?;

    // Convert ClientCommandResponse to SubmitCommandResponse
    match result {
        crate::response::ClientCommandResponse::Transaction(tx) => {
            Ok(SubmitCommandResponse::Transaction(tx))
        }
        crate::response::ClientCommandResponse::SerializedUnsignedTransaction(s) => {
            Ok(SubmitCommandResponse::SerializedTransaction { serialized_transaction: s })
        }
        crate::response::ClientCommandResponse::SerializedSignedTransaction(s) => {
            Ok(SubmitCommandResponse::SerializedTransaction { serialized_transaction: s })
        }
        crate::response::ClientCommandResponse::TransactionDigest(d) => {
            Ok(SubmitCommandResponse::TransactionDigest(d))
        }
        crate::response::ClientCommandResponse::Simulation(sim) => {
            Ok(SubmitCommandResponse::Simulation(sim))
        }
        _ => bail!("Unexpected response type from transaction execution"),
    }
}

// =============================================================================
// Response types
// =============================================================================

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum SubmitCommandResponse {
    SubmitSuccess {
        #[serde(flatten)]
        transaction: TransactionResponse,
        #[serde(skip_serializing_if = "Option::is_none")]
        next_epoch_hint: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        claim_epoch_hint: Option<String>,
    },
    Transaction(TransactionResponse),
    SerializedTransaction {
        serialized_transaction: String,
    },
    TransactionDigest(types::digests::TransactionDigest),
    Simulation(crate::response::SimulationResponse),
}

impl Display for SubmitCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SubmitCommandResponse::SubmitSuccess {
                transaction,
                next_epoch_hint,
                claim_epoch_hint,
            } => {
                write!(f, "{}", transaction)?;
                writeln!(f)?;
                writeln!(f, "  {}", "Submission accepted.".green().bold())?;
                if let Some(hint) = next_epoch_hint {
                    writeln!(
                        f,
                        "  {} Challenge window closes at epoch boundary ({})",
                        ">>".dimmed(),
                        hint.yellow()
                    )?;
                }
                if let Some(hint) = claim_epoch_hint {
                    writeln!(f, "  {} {}", ">>".dimmed(), hint.yellow())?;
                }
                Ok(())
            }
            SubmitCommandResponse::Transaction(tx_response) => {
                write!(f, "{}", tx_response)
            }
            SubmitCommandResponse::SerializedTransaction { serialized_transaction } => {
                writeln!(f, "{}", "Serialized Transaction".cyan().bold())?;
                writeln!(f)?;
                writeln!(f, "{}", serialized_transaction)?;
                writeln!(f)?;
                writeln!(f, "{}", "Use 'soma tx execute-signed' to submit after signing.".yellow())
            }
            SubmitCommandResponse::TransactionDigest(digest) => {
                writeln!(f, "{}: {}", "Transaction Digest".bold(), digest)
            }
            SubmitCommandResponse::Simulation(sim) => write!(f, "{}", sim),
        }
    }
}

impl SubmitCommandResponse {
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
