// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use super::Client;
use crate::proto::TryFromProtoError;
use crate::proto::soma::ExecuteTransactionRequest;
use crate::proto::soma::ExecuteTransactionResponse;
use crate::proto::soma::ExecutionError;
use crate::proto::soma::GetEpochRequest;
use crate::proto::soma::SubscribeCheckpointsRequest;
use crate::utils::field::FieldMaskUtil;
use futures::TryStreamExt;
use prost_types::FieldMask;
use std::fmt;
use std::time::Duration;
use tonic::Response;

/// Response from execute_transaction_and_wait_for_checkpoint
pub struct ExecuteAndWaitResponse {
    pub response: Response<ExecuteTransactionResponse>,
    pub checkpoint_sequence_number: u64,
}

/// Error types that can occur when executing a transaction and waiting for checkpoint
#[derive(Debug)]
#[non_exhaustive]
pub enum ExecuteAndWaitError {
    /// RPC Error (actual tonic::Status from the client/server)
    RpcError(tonic::Status),
    /// Request is missing the required transaction field
    MissingTransaction,
    /// Failed to parse/convert the transaction for digest calculation
    ProtoConversionError(TryFromProtoError),
    /// Transaction executed but checkpoint wait timed out
    CheckpointTimeout(Response<ExecuteTransactionResponse>),
    /// Transaction executed but checkpoint stream had an error
    CheckpointStreamError { response: Response<ExecuteTransactionResponse>, error: tonic::Status },
}

impl std::fmt::Display for ExecuteAndWaitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RpcError(status) => write!(f, "RPC error: {status}"),
            Self::MissingTransaction => {
                write!(f, "Request is missing the required transaction field")
            }
            Self::ProtoConversionError(e) => write!(f, "Failed to convert transaction: {e}"),
            Self::CheckpointTimeout(_) => {
                write!(f, "Transaction executed but checkpoint wait timed out")
            }
            Self::CheckpointStreamError { error, .. } => {
                write!(f, "Transaction executed but checkpoint stream had an error: {error}")
            }
        }
    }
}

impl std::error::Error for ExecuteAndWaitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RpcError(status) => Some(status),
            Self::ProtoConversionError(e) => Some(e),
            Self::CheckpointStreamError { error, .. } => Some(error),
            Self::MissingTransaction => None,
            Self::CheckpointTimeout(_) => None,
        }
    }
}

impl Client {
    /// Executes a transaction and waits for it to be included in a checkpoint.
    ///
    /// This method provides "read your writes" consistency by executing the transaction
    /// and waiting for it to appear in a checkpoint, which gauruntees indexes have been updated on
    /// this node.
    ///
    /// # Arguments
    /// * `request` - The transaction execution request (ExecuteTransactionRequest)
    /// * `timeout` - Maximum time to wait for indexing confirmation
    ///
    /// # Returns
    /// A `Result` containing the response if the transaction was executed and checkpoint confirmed,
    /// or an error that may still include the response if execution succeeded but checkpoint
    /// confirmation failed.
    pub async fn execute_transaction_and_wait_for_checkpoint(
        &mut self,
        request: impl tonic::IntoRequest<ExecuteTransactionRequest>,
        timeout: Duration,
    ) -> Result<ExecuteAndWaitResponse, ExecuteAndWaitError> {
        // Subscribe to checkpoint stream before execution to avoid missing the transaction.
        // Uses minimal read mask for efficiency since we only need digest confirmation.
        // Once server-side filtering is available, we should filter by transaction digest to
        // further reduce bandwidth.
        let mut checkpoint_stream = match self
            .subscription_client()
            .subscribe_checkpoints(
                SubscribeCheckpointsRequest::default()
                    .with_read_mask(FieldMask::from_str("transactions.digest,sequence_number")),
            )
            .await
        {
            Ok(stream) => stream.into_inner(),
            Err(e) => return Err(ExecuteAndWaitError::RpcError(e)),
        };

        // Calculate digest from the input transaction to avoid relying on response read mask
        let request = request.into_request();
        let transaction = match request.get_ref().transaction_opt() {
            Some(tx) => tx,
            None => return Err(ExecuteAndWaitError::MissingTransaction),
        };

        let executed_txn_digest = transaction.digest.clone().ok_or_else(|| {
            ExecuteAndWaitError::ProtoConversionError(TryFromProtoError::missing("digest"))
        })?;

        let response = match self.execution_client().execute_transaction(request).await {
            Ok(resp) => resp,
            Err(e) => return Err(ExecuteAndWaitError::RpcError(e)),
        };

        // Wait for the transaction to appear in a checkpoint, at which point indexes will have been
        // updated.
        let timeout_future = tokio::time::sleep(timeout);
        let checkpoint_future = async {
            while let Some(checkpoint_response) = checkpoint_stream.try_next().await? {
                let checkpoint = checkpoint_response.checkpoint();
                let sequence_number = checkpoint_response.cursor.unwrap_or(0);

                for tx in checkpoint.transactions() {
                    let digest = tx.digest();

                    if digest == executed_txn_digest {
                        return Ok(sequence_number);
                    }
                }
            }
            Err(tonic::Status::aborted("checkpoint stream ended unexpectedly"))
        };

        tokio::select! {
            result = checkpoint_future => {
                match result {
                    Ok(checkpoint_sequence_number) => Ok(ExecuteAndWaitResponse {
                        response,
                        checkpoint_sequence_number,
                    }),
                    Err(e) => Err(ExecuteAndWaitError::CheckpointStreamError { response, error: e })
                }
            },
            _ = timeout_future => {
                Err(ExecuteAndWaitError::CheckpointTimeout(response))
            }
        }
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = self.description.as_deref().unwrap_or("No description");
        write!(
            f,
            "ExecutionError: Kind: {}, Description: {}",
            self.kind().as_str_name(),
            description
        )
    }
}
