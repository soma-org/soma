// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use rpc::api::error::RpcError;
use rpc::proto::google::rpc::bad_request::FieldViolation;
use rpc::proto::soma::{
    BatchGetTransactionsRequest, BatchGetTransactionsResponse, ErrorReason, ExecutedTransaction,
    GetTransactionRequest, GetTransactionResponse, GetTransactionResult, Transaction,
    TransactionEffects, UserSignature,
};
use rpc::proto::timestamp_ms_to_proto;
use rpc::types::Digest;
use rpc::utils::field::{FieldMask, FieldMaskTree, FieldMaskUtil};
use rpc::utils::merge::Merge;

use crate::KeyValueStoreReader;
use crate::TransactionData;

use super::KvRpcServer;

pub const READ_MASK_DEFAULT: &str = "digest";

pub async fn get_transaction(
    server: &KvRpcServer,
    request: GetTransactionRequest,
) -> Result<GetTransactionResponse, RpcError> {
    let transaction_digest = request
        .digest
        .ok_or_else(|| {
            FieldViolation::new("digest")
                .with_description("missing digest")
                .with_reason(ErrorReason::FieldMissing)
        })?
        .parse::<Digest>()
        .map_err(|e| {
            FieldViolation::new("digest")
                .with_description(format!("invalid digest: {e}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;

    let read_mask = {
        let read_mask = request.read_mask.unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
        read_mask.validate::<ExecutedTransaction>().map_err(|path| {
            FieldViolation::new("read_mask")
                .with_description(format!("invalid read_mask path: {path}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;
        FieldMaskTree::from(read_mask)
    };

    let mut client = server.client.clone();
    let core_digest = types::digests::TransactionDigest::new(transaction_digest.into_inner());

    let mut results = client.get_transactions(&[core_digest]).await?;
    let tx_data = results.pop().ok_or_else(|| {
        RpcError::new(rpc_tonic::Code::NotFound, format!("Transaction {core_digest} not found"))
    })?;

    let transaction = transaction_to_response(tx_data, &read_mask)?;
    Ok(GetTransactionResponse::new(transaction))
}

pub async fn batch_get_transactions(
    server: &KvRpcServer,
    BatchGetTransactionsRequest { digests, read_mask, .. }: BatchGetTransactionsRequest,
) -> Result<BatchGetTransactionsResponse, RpcError> {
    let read_mask = {
        let read_mask = read_mask.unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
        read_mask.validate::<ExecutedTransaction>().map_err(|path| {
            FieldViolation::new("read_mask")
                .with_description(format!("invalid read_mask path: {path}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;
        FieldMaskTree::from(read_mask)
    };

    let mut client = server.client.clone();

    let transactions: Vec<GetTransactionResult> = {
        let mut results = Vec::with_capacity(digests.len());
        for (idx, digest_str) in digests.into_iter().enumerate() {
            let result = async {
                let digest = digest_str.parse::<Digest>().map_err(|e| {
                    RpcError::from(
                        FieldViolation::new_at("digests", idx)
                            .with_description(format!("invalid digest: {e}"))
                            .with_reason(ErrorReason::FieldInvalid),
                    )
                })?;
                let core_digest = types::digests::TransactionDigest::new(digest.into_inner());
                let mut tx_results = client.get_transactions(&[core_digest]).await?;
                let tx_data = tx_results.pop().ok_or_else(|| {
                    RpcError::new(
                        rpc_tonic::Code::NotFound,
                        format!("Transaction {core_digest} not found"),
                    )
                })?;
                transaction_to_response(tx_data, &read_mask)
            }
            .await;

            results.push(match result {
                Ok(transaction) => GetTransactionResult::new_transaction(transaction),
                Err(error) => GetTransactionResult::new_error(error.into_status_proto()),
            });
        }
        results
    };

    Ok(BatchGetTransactionsResponse::new(transactions))
}

fn transaction_to_response(
    source: TransactionData,
    mask: &FieldMaskTree,
) -> Result<ExecutedTransaction, RpcError> {
    let mut message = ExecutedTransaction::default();

    // Convert core types to SDK types
    let sdk_signed_tx: rpc::types::SignedTransaction =
        source.transaction.clone().try_into().map_err(|e| {
            RpcError::new(rpc_tonic::Code::Internal, format!("failed to convert transaction: {e}"))
        })?;

    let sdk_effects: rpc::types::TransactionEffects =
        source.effects.clone().try_into().map_err(|e| {
            RpcError::new(rpc_tonic::Code::Internal, format!("failed to convert effects: {e}"))
        })?;

    if mask.contains(ExecutedTransaction::DIGEST_FIELD.name) {
        message.digest = Some(source.transaction.digest().to_string());
    }

    if let Some(submask) = mask.subtree(ExecutedTransaction::TRANSACTION_FIELD.name) {
        message.transaction = Some(Transaction::merge_from(sdk_signed_tx.transaction, &submask));
    }

    if let Some(submask) = mask.subtree(ExecutedTransaction::SIGNATURES_FIELD.name) {
        message.signatures = sdk_signed_tx
            .signatures
            .into_iter()
            .map(|s| UserSignature::merge_from(s, &submask))
            .collect();
    }

    if let Some(submask) = mask.subtree(ExecutedTransaction::EFFECTS_FIELD.name) {
        message.effects = Some(TransactionEffects::merge_from(&sdk_effects, &submask));
    }

    if mask.contains(ExecutedTransaction::CHECKPOINT_FIELD.name) {
        message.checkpoint = Some(source.checkpoint_number);
    }

    if mask.contains(ExecutedTransaction::TIMESTAMP_FIELD.name) {
        message.timestamp = Some(timestamp_ms_to_proto(source.timestamp));
    }

    Ok(message)
}
