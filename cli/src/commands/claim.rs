//! Claim command for claiming rewards from filled targets (`soma target claim`).

use anyhow::{Result, bail};
use clap::*;
use colored::Colorize;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};

use sdk::wallet_context::WalletContext;
use types::{
    object::ObjectID,
    transaction::{ClaimRewardsArgs, TransactionKind},
};

use crate::client_commands::TxProcessingArgs;
use crate::response::TransactionResponse;

/// Claim rewards from a filled target.
///
/// Claims the reward pool from a target that was filled by the sender.
/// The challenge window (one full epoch after the target was filled) must have closed.
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub struct ClaimCommand {
    /// Target ID to claim rewards from
    pub target_id: ObjectID,
    #[clap(flatten)]
    pub tx_args: TxProcessingArgs,
}

impl ClaimCommand {
    pub async fn execute(self, context: &mut WalletContext) -> Result<ClaimCommandResponse> {
        let sender = context.active_address()?;

        let kind = TransactionKind::ClaimRewards(ClaimRewardsArgs { target_id: self.target_id });

        let result =
            crate::client_commands::execute_or_serialize(context, sender, kind, None, self.tx_args)
                .await?;

        // Convert ClientCommandResponse to ClaimCommandResponse
        match result {
            crate::response::ClientCommandResponse::Transaction(tx) => {
                Ok(ClaimCommandResponse::Transaction(tx))
            }
            crate::response::ClientCommandResponse::SerializedUnsignedTransaction(s) => {
                Ok(ClaimCommandResponse::SerializedTransaction { serialized_transaction: s })
            }
            crate::response::ClientCommandResponse::SerializedSignedTransaction(s) => {
                Ok(ClaimCommandResponse::SerializedTransaction { serialized_transaction: s })
            }
            crate::response::ClientCommandResponse::TransactionDigest(d) => {
                Ok(ClaimCommandResponse::TransactionDigest(d))
            }
            crate::response::ClientCommandResponse::Simulation(sim) => {
                Ok(ClaimCommandResponse::Simulation(sim))
            }
            _ => bail!("Unexpected response type from transaction execution"),
        }
    }
}

// =============================================================================
// Response types
// =============================================================================

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ClaimCommandResponse {
    Transaction(TransactionResponse),
    SerializedTransaction { serialized_transaction: String },
    TransactionDigest(types::digests::TransactionDigest),
    Simulation(crate::response::SimulationResponse),
}

impl Display for ClaimCommandResponse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ClaimCommandResponse::Transaction(tx_response) => {
                write!(f, "{}", tx_response)
            }
            ClaimCommandResponse::SerializedTransaction { serialized_transaction } => {
                writeln!(f, "{}", "Serialized Transaction".cyan().bold())?;
                writeln!(f)?;
                writeln!(f, "{}", serialized_transaction)?;
                writeln!(f)?;
                writeln!(
                    f,
                    "{}",
                    "Use 'soma tx execute-signed' to submit after signing.".yellow()
                )
            }
            ClaimCommandResponse::TransactionDigest(digest) => {
                writeln!(f, "{}: {}", "Transaction Digest".bold(), digest)
            }
            ClaimCommandResponse::Simulation(sim) => write!(f, "{}", sim),
        }
    }
}

impl ClaimCommandResponse {
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
