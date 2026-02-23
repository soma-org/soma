// Copyright (c) Mysten Labs, Inc.
// Portions of this file are derived from Sui (https://github.com/MystenLabs/sui).
// SPDX-License-Identifier: Apache-2.0

use crate::api::RpcService;
use crate::api::error::RpcError;
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::BatchGetTransactionsRequest;
use crate::proto::soma::BatchGetTransactionsResponse;
use crate::proto::soma::ErrorReason;
use crate::proto::soma::ExecutedTransaction;
use crate::proto::soma::GetTransactionRequest;
use crate::proto::soma::GetTransactionResponse;
use crate::proto::soma::GetTransactionResult;
use crate::proto::soma::Transaction;
use crate::proto::soma::TransactionEffects;
use crate::proto::soma::UserSignature;
use crate::proto::timestamp_ms_to_proto;
use crate::types::{Address, Digest};
use crate::utils::field::FieldMask;
use crate::utils::field::FieldMaskTree;
use crate::utils::field::FieldMaskUtil;
use crate::utils::merge::Merge;

pub const READ_MASK_DEFAULT: &str = "digest";

#[tracing::instrument(skip(service))]
pub fn get_transaction(
    service: &RpcService,
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

    let transaction_read = service.reader.get_transaction_read(transaction_digest)?;

    let transaction = transaction_to_response(service, transaction_read, &read_mask);

    Ok(GetTransactionResponse::new(transaction))
}

#[tracing::instrument(skip(service))]
pub fn batch_get_transactions(
    service: &RpcService,
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

    let transactions = digests
        .into_iter()
        .enumerate()
        .map(|(idx, digest)| {
            let digest = digest.parse().map_err(|e| {
                FieldViolation::new_at("digests", idx)
                    .with_description(format!("invalid digest: {e}"))
                    .with_reason(ErrorReason::FieldInvalid)
            })?;

            service.reader.get_transaction_read(digest).map(|transaction_read| {
                transaction_to_response(service, transaction_read, &read_mask)
            })
        })
        .map(|result| match result {
            Ok(transaction) => GetTransactionResult::new_transaction(transaction),
            Err(error) => GetTransactionResult::new_error(error.into_status_proto()),
        })
        .collect();

    Ok(BatchGetTransactionsResponse::new(transactions))
}

fn transaction_to_response(
    service: &RpcService,
    source: crate::api::reader::TransactionRead,
    mask: &FieldMaskTree,
) -> ExecutedTransaction {
    let mut message = ExecutedTransaction::default();

    if mask.contains(ExecutedTransaction::DIGEST_FIELD.name) {
        message.digest = Some(source.digest.to_string());
    }

    if let Some(submask) = mask.subtree(ExecutedTransaction::TRANSACTION_FIELD.name) {
        message.transaction = Some(Transaction::merge_from(source.transaction, &submask));
    }

    if let Some(submask) = mask.subtree(ExecutedTransaction::SIGNATURES_FIELD.name) {
        message.signatures =
            source.signatures.into_iter().map(|s| UserSignature::merge_from(s, &submask)).collect();
    }

    if let Some(submask) = mask.subtree(ExecutedTransaction::EFFECTS_FIELD.name) {
        let mut effects = TransactionEffects::merge_from(&source.effects, &submask);

        if let Some(object_types) = source.object_types {
            if submask.contains(TransactionEffects::CHANGED_OBJECTS_FIELD.name) {
                for changed_object in effects.changed_objects.iter_mut() {
                    let Ok(object_id) = changed_object.object_id().parse::<Address>() else {
                        continue;
                    };

                    if let Some(ty) = object_types.get(&object_id.into()) {
                        changed_object.object_type = Some(ty.to_string());
                    }
                }
            }

            if submask.contains(TransactionEffects::UNCHANGED_SHARED_OBJECTS_FIELD.name) {
                for unchanged_shared_object in effects.unchanged_shared_objects.iter_mut() {
                    let Ok(object_id) = unchanged_shared_object.object_id().parse::<Address>()
                    else {
                        continue;
                    };

                    if let Some(ty) = object_types.get(&object_id.into()) {
                        unchanged_shared_object.object_type = Some(ty.to_string());
                    }
                }
            }
        }

        message.effects = Some(effects);
    }

    if mask.contains(ExecutedTransaction::CHECKPOINT_FIELD.name) {
        message.checkpoint = source.checkpoint;
    }

    if mask.contains(ExecutedTransaction::TIMESTAMP_FIELD.name) {
        message.timestamp = source.timestamp_ms.map(timestamp_ms_to_proto);
    }

    if mask.contains(ExecutedTransaction::BALANCE_CHANGES_FIELD.name) {
        message.balance_changes = source
            .balance_changes
            .map(|balance_changes| balance_changes.into_iter().map(Into::into).collect())
            .unwrap_or_default();
    }

    message
}
