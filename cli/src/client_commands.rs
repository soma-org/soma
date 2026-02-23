//! Advanced client commands for power users and backward compatibility.
//!
//! Most common operations have been promoted to top-level commands (balance, send, transfer, etc.).
//! This module contains advanced operations like executing serialized transactions.

use anyhow::{Result, anyhow, bail, ensure};
use clap::*;
use fastcrypto::encoding::{Base64, Encoding};
use fastcrypto::traits::ToFromBytes;
use types::effects::{ExecutionStatus, TransactionEffectsAPI as _};

use sdk::{transaction_builder::TransactionBuilder, wallet_context::WalletContext};
use types::{
    base::SomaAddress,
    crypto::GenericSignature,
    envelope::Envelope,
    object::ObjectRef,
    transaction::{SenderSignedData, Transaction, TransactionData, TransactionKind},
};

use crate::response::{
    ClientCommandResponse, SimulationResponse, TransactionResponse, TransactionStatus,
};

/// Arguments related to transaction processing
#[derive(Args, Debug, Default, Clone)]
pub struct TxProcessingArgs {
    /// Compute the transaction digest and print it out, but do not execute.
    #[arg(long)]
    pub tx_digest: bool,

    /// Perform a dry run (simulation) of the transaction, without executing it.
    #[arg(long)]
    pub simulate: bool,

    /// Serialize the unsigned transaction data (base64) instead of executing.
    /// Use with `soma client execute-signed-tx --tx-bytes <TX_BYTES>`.
    #[arg(long)]
    pub serialize_unsigned_transaction: bool,

    /// Serialize the signed transaction data (base64) instead of executing.
    /// Use with `soma client execute-combined-signed-tx --signed-tx-bytes <SIGNED_TX_BYTES>`.
    #[arg(long)]
    pub serialize_signed_transaction: bool,
}

/// Advanced client commands (for backward compatibility and power users)
#[derive(Parser)]
#[clap(rename_all = "kebab-case")]
pub enum SomaClientCommands {
    /// Execute from serialized transaction bytes
    #[clap(name = "execute-serialized")]
    ExecuteSerialized {
        /// Base64-encoded BCS-serialized TransactionData
        tx_bytes: String,
        #[clap(flatten)]
        processing: TxProcessingArgs,
    },

    /// Execute using pre-signed transaction bytes and signatures
    #[clap(name = "execute-signed-tx")]
    ExecuteSignedTx {
        /// Base64-encoded unsigned transaction data
        #[clap(long)]
        tx_bytes: String,
        /// Base64-encoded signatures (flag || signature || pubkey)
        #[clap(long)]
        signatures: Vec<String>,
    },

    /// Execute a combined sender-signed transaction
    #[clap(name = "execute-combined-signed-tx")]
    ExecuteCombinedSignedTx {
        /// Base64-encoded SenderSignedData
        #[clap(long)]
        signed_tx_bytes: String,
    },
}

impl SomaClientCommands {
    pub async fn execute(self, context: &mut WalletContext) -> Result<ClientCommandResponse> {
        match self {
            SomaClientCommands::ExecuteSerialized { tx_bytes, processing } => {
                let bytes =
                    Base64::decode(&tx_bytes).map_err(|_| anyhow!("Invalid Base64 encoding"))?;

                let tx_data: TransactionData = bcs::from_bytes(&bytes)
                    .map_err(|_| anyhow!("Failed to parse TransactionData"))?;

                let sender = tx_data.sender();
                let kind = tx_data.kind().clone();
                let gas = tx_data.gas().first().cloned();

                execute_or_serialize(context, sender, kind, gas, processing).await
            }

            SomaClientCommands::ExecuteSignedTx { tx_bytes, signatures } => {
                let data_bytes = Base64::decode(&tx_bytes)
                    .map_err(|_| anyhow!("Invalid Base64 encoding for tx_bytes"))?;

                let data: TransactionData = bcs::from_bytes(&data_bytes).map_err(|_| {
                    anyhow!(
                        "Failed to parse tx_bytes. \
                         Ensure it matches output from --serialize-unsigned-transaction"
                    )
                })?;

                let mut sigs = Vec::new();
                for sig_str in signatures {
                    let sig_bytes = Base64::decode(&sig_str)
                        .map_err(|_| anyhow!("Invalid Base64 encoding for signature"))?;
                    let sig = GenericSignature::from_bytes(&sig_bytes)
                        .map_err(|_| anyhow!("Invalid signature format"))?;
                    sigs.push(sig);
                }

                let transaction = Transaction::from_generic_sig_data(data, sigs);
                let response = context.execute_transaction_may_fail(transaction).await?;

                Ok(ClientCommandResponse::Transaction(TransactionResponse::from_response(
                    &response,
                )))
            }

            SomaClientCommands::ExecuteCombinedSignedTx { signed_tx_bytes } => {
                let bytes = Base64::decode(&signed_tx_bytes)
                    .map_err(|_| anyhow!("Invalid Base64 encoding"))?;

                let sender_signed: SenderSignedData = bcs::from_bytes(&bytes).map_err(|_| {
                    anyhow!(
                        "Failed to parse SenderSignedData. \
                         Ensure it matches output from --serialize-signed-transaction"
                    )
                })?;

                let transaction =
                    Envelope::<SenderSignedData, types::crypto::EmptySignInfo>::new(sender_signed);
                let response = context.execute_transaction_may_fail(transaction).await?;

                Ok(ClientCommandResponse::Transaction(TransactionResponse::from_response(
                    &response,
                )))
            }
        }
    }
}

/// Execute a transaction or serialize it based on processing args.
/// This is a shared helper used by multiple command modules.
pub async fn execute_or_serialize(
    context: &mut WalletContext,
    sender: SomaAddress,
    kind: TransactionKind,
    gas: Option<ObjectRef>,
    processing: TxProcessingArgs,
) -> Result<ClientCommandResponse> {
    ensure!(
        !(processing.serialize_unsigned_transaction && processing.serialize_signed_transaction),
        "Cannot specify both --serialize-unsigned-transaction and --serialize-signed-transaction"
    );

    // Build transaction data
    let builder = TransactionBuilder::new(context);
    let tx_data = builder.build_transaction_data(sender, kind, gas).await?;

    // Handle tx-digest-only mode
    if processing.tx_digest {
        return Ok(ClientCommandResponse::TransactionDigest(tx_data.digest()));
    }

    // Handle simulation mode (no signature required)
    if processing.simulate {
        let client = context.get_client().await?;
        let result = client
            .simulate_transaction(&tx_data)
            .await
            .map_err(|e| anyhow!("Simulation failed: {}", e))?;

        let status = match result.effects.status() {
            ExecutionStatus::Success => TransactionStatus::Success,
            ExecutionStatus::Failure { error } => {
                TransactionStatus::Failure { error: format!("{}", error) }
            }
        };

        return Ok(ClientCommandResponse::Simulation(SimulationResponse {
            status,
            gas_used: result.effects.transaction_fee().total_fee,
            created: result.effects.created().into_iter().map(Into::into).collect(),
            mutated: result.effects.mutated_excluding_gas().into_iter().map(Into::into).collect(),
            deleted: result.effects.deleted().into_iter().map(Into::into).collect(),
            balance_changes: result.balance_changes,
        }));
    }

    // Handle serialize-unsigned mode
    if processing.serialize_unsigned_transaction {
        let bytes = bcs::to_bytes(&tx_data)?;
        let encoded = Base64::encode(&bytes);
        return Ok(ClientCommandResponse::SerializedUnsignedTransaction(encoded));
    }

    // Sign the transaction
    let tx = context.sign_transaction(&tx_data).await;

    // Handle serialize-signed mode
    if processing.serialize_signed_transaction {
        let bytes = bcs::to_bytes(tx.data())?;
        let encoded = Base64::encode(&bytes);
        return Ok(ClientCommandResponse::SerializedSignedTransaction(encoded));
    }

    // Execute the transaction
    let response = context.execute_transaction_may_fail(tx).await?;

    Ok(ClientCommandResponse::Transaction(TransactionResponse::from_response(&response)))
}
