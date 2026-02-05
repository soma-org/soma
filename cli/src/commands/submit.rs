use anyhow::{Result, anyhow, bail};
use clap::*;
use colored::Colorize;
use fastcrypto::encoding::{Encoding, Hex};
use serde::Serialize;
use std::fmt::{self, Display, Formatter};

use sdk::wallet_context::WalletContext;
use types::{
    base::SomaAddress,
    checksum::Checksum,
    digests::DataCommitment,
    metadata::{Manifest, ManifestV1, Metadata, MetadataV1},
    object::ObjectID,
    submission::SubmissionManifest,
    target::Embedding,
    transaction::{SubmitDataArgs, TransactionKind},
};

use crate::client_commands::TxProcessingArgs;
use crate::response::TransactionResponse;

/// Submit data to fill a target
///
/// Submits data that embeds within a target's radius. The submission
/// includes the data manifest, embedding vector, and reported scores.
/// A bond (proportional to data size) is required.
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub struct SubmitCommand {
    /// Target ID to submit to
    #[clap(long)]
    pub target_id: ObjectID,
    /// Hex-encoded commitment to the data (32 bytes, hash of raw data)
    #[clap(long)]
    pub data_commitment: String,
    /// URL where the data is hosted
    #[clap(long)]
    pub data_url: String,
    /// Hex-encoded checksum of the data file (32 bytes)
    #[clap(long)]
    pub data_checksum: String,
    /// Size of the data file in bytes
    #[clap(long)]
    pub data_size: usize,
    /// Model ID to use for this submission (must be in target's model_ids)
    #[clap(long)]
    pub model_id: ObjectID,
    /// Embedding vector (comma-separated fixed-point i64 values)
    #[clap(long)]
    pub embedding: String,
    /// Distance score (fixed-point i64, must be <= target threshold)
    #[clap(long)]
    pub distance_score: i64,
    /// Reconstruction score (fixed-point u64, must be <= target threshold)
    #[clap(long)]
    pub reconstruction_score: u64,
    /// Coin object to use for bond payment
    #[clap(long)]
    pub bond_coin: ObjectID,
    #[clap(flatten)]
    pub tx_args: TxProcessingArgs,
}

// =============================================================================
// Execution
// =============================================================================

impl SubmitCommand {
    pub async fn execute(
        self,
        context: &mut WalletContext,
    ) -> Result<SubmitCommandResponse> {
        let sender = context.active_address()?;
        let client = context.get_client().await?;

        // Parse data commitment
        let commitment_bytes = parse_hex_digest_32(&self.data_commitment, "data-commitment")?;

        // Build data manifest
        let manifest = build_data_manifest(&self.data_url, &self.data_checksum, self.data_size)?;

        // Parse embedding
        let embedding_vec = parse_embedding(&self.embedding)?;

        // Get bond coin reference
        let coin_obj = client
            .get_object(self.bond_coin)
            .await
            .map_err(|e| anyhow!("Failed to get bond coin: {}", e))?;
        let bond_coin_ref = coin_obj.compute_object_reference();

        let kind = TransactionKind::SubmitData(SubmitDataArgs {
            target_id: self.target_id,
            data_commitment: DataCommitment::new(commitment_bytes),
            data_manifest: manifest,
            model_id: self.model_id,
            embedding: Embedding::from(embedding_vec),
            distance_score: self.distance_score,
            reconstruction_score: self.reconstruction_score,
            bond_coin: bond_coin_ref,
        });

        execute_tx(context, sender, kind, self.tx_args).await
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn parse_hex_digest_32(hex_str: &str, field_name: &str) -> Result<[u8; 32]> {
    let bytes = Hex::decode(hex_str.strip_prefix("0x").unwrap_or(hex_str))
        .map_err(|e| anyhow!("Invalid hex for {}: {}", field_name, e))?;
    let arr: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow!("{} must be exactly 32 bytes", field_name))?;
    Ok(arr)
}

fn build_data_manifest(
    url: &str,
    checksum_hex: &str,
    size: usize,
) -> Result<SubmissionManifest> {
    let parsed_url: url::Url = url.parse().map_err(|e| anyhow!("Invalid URL: {}", e))?;
    let checksum_bytes = parse_hex_digest_32(checksum_hex, "data-checksum")?;

    let metadata = Metadata::V1(MetadataV1::new(Checksum(checksum_bytes), size));
    let manifest = Manifest::V1(ManifestV1::new(parsed_url, metadata));

    Ok(SubmissionManifest::new(manifest))
}

fn parse_embedding(embedding_str: &str) -> Result<Vec<i64>> {
    embedding_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<i64>()
                .map_err(|e| anyhow!("Invalid embedding value '{}': {}", s, e))
        })
        .collect()
}

/// Execute a submit transaction, delegating to the shared client_commands helper.
async fn execute_tx(
    context: &mut WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
    tx_args: TxProcessingArgs,
) -> Result<SubmitCommandResponse> {
    let result = crate::client_commands::execute_or_serialize(
        context, sender, kind, None, tx_args,
    )
    .await?;

    // Convert ClientCommandResponse to SubmitCommandResponse
    match result {
        crate::response::ClientCommandResponse::Transaction(tx) => {
            Ok(SubmitCommandResponse::Transaction(tx))
        }
        crate::response::ClientCommandResponse::SerializedUnsignedTransaction(s) => {
            Ok(SubmitCommandResponse::SerializedTransaction {
                serialized_unsigned_transaction: s,
            })
        }
        crate::response::ClientCommandResponse::SerializedSignedTransaction(s) => {
            Ok(SubmitCommandResponse::SerializedTransaction {
                serialized_unsigned_transaction: s,
            })
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
    Transaction(TransactionResponse),
    SerializedTransaction {
        serialized_unsigned_transaction: String,
    },
    TransactionDigest(types::digests::TransactionDigest),
    Simulation(crate::response::SimulationResponse),
}

impl Display for SubmitCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SubmitCommandResponse::Transaction(tx_response) => {
                write!(f, "{}", tx_response)
            }
            SubmitCommandResponse::SerializedTransaction {
                serialized_unsigned_transaction,
            } => {
                writeln!(f, "{}", "Serialized Unsigned Transaction".cyan().bold())?;
                writeln!(f)?;
                writeln!(f, "{}", serialized_unsigned_transaction)?;
                writeln!(f)?;
                writeln!(
                    f,
                    "{}",
                    "Use 'soma client execute-signed-tx' to submit after signing.".yellow()
                )
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
