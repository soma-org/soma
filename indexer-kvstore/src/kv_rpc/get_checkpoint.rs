// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

use rpc::api::error::{CheckpointNotFoundError, RpcError};
use rpc::proto::google::rpc::bad_request::FieldViolation;
use rpc::proto::soma::get_checkpoint_request::CheckpointId;
use rpc::proto::soma::{Checkpoint, ErrorReason, GetCheckpointRequest, GetCheckpointResponse};
use rpc::types::Digest;
use rpc::utils::field::{FieldMask, FieldMaskTree, FieldMaskUtil};
use rpc::utils::merge::Merge;

use crate::KeyValueStoreReader;

use super::KvRpcServer;

pub const READ_MASK_DEFAULT: &str = "sequence_number,digest";

pub async fn get_checkpoint(
    server: &KvRpcServer,
    request: GetCheckpointRequest,
) -> Result<GetCheckpointResponse, RpcError> {
    let read_mask = {
        let read_mask = request
            .read_mask
            .unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
        read_mask.validate::<Checkpoint>().map_err(|path| {
            FieldViolation::new("read_mask")
                .with_description(format!("invalid read_mask path: {path}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;
        FieldMaskTree::from(read_mask)
    };

    let mut client = server.client.clone();

    let checkpoint_data = match request.checkpoint_id {
        Some(CheckpointId::SequenceNumber(s)) => {
            let mut results = client.get_checkpoints(&[s]).await?;
            results
                .pop()
                .ok_or(CheckpointNotFoundError::sequence_number(s))?
        }
        Some(CheckpointId::Digest(digest_str)) => {
            let digest = digest_str.parse::<Digest>().map_err(|e| {
                FieldViolation::new("digest")
                    .with_description(format!("invalid digest: {e}"))
                    .with_reason(ErrorReason::FieldInvalid)
            })?;

            let core_digest = types::digests::CheckpointDigest::new(digest.into_inner());
            client
                .get_checkpoint_by_digest(core_digest)
                .await?
                .ok_or(CheckpointNotFoundError::digest(digest))?
        }
        None => {
            // Latest checkpoint: use watermark to find the highest sequence number
            let wm = client
                .get_watermark()
                .await?
                .ok_or_else(RpcError::not_found)?;
            let mut results = client.get_checkpoints(&[wm.checkpoint_hi_inclusive]).await?;
            results
                .pop()
                .ok_or(CheckpointNotFoundError::sequence_number(
                    wm.checkpoint_hi_inclusive,
                ))?
        }
        _ => {
            // Future checkpoint_id variants - treat as latest
            let wm = client
                .get_watermark()
                .await?
                .ok_or_else(RpcError::not_found)?;
            let mut results = client.get_checkpoints(&[wm.checkpoint_hi_inclusive]).await?;
            results
                .pop()
                .ok_or(CheckpointNotFoundError::sequence_number(
                    wm.checkpoint_hi_inclusive,
                ))?
        }
    };

    // Convert core types to SDK types
    let sdk_summary: rpc::types::CheckpointSummary =
        checkpoint_data.summary.clone().try_into().map_err(|e| {
            RpcError::new(
                rpc_tonic::Code::Internal,
                format!("failed to convert checkpoint summary: {e}"),
            )
        })?;

    let sdk_signature: rpc::types::ValidatorAggregatedSignature =
        checkpoint_data.signatures.into();

    let sdk_contents: rpc::types::CheckpointContents =
        checkpoint_data.contents.try_into().map_err(|e| {
            RpcError::new(
                rpc_tonic::Code::Internal,
                format!("failed to convert checkpoint contents: {e}"),
            )
        })?;

    // Build proto response via Merge
    let mut checkpoint = Checkpoint::default();
    checkpoint.merge(&sdk_summary, &read_mask);
    checkpoint.merge(sdk_signature, &read_mask);

    if read_mask.contains(Checkpoint::CONTENTS_FIELD.name) {
        checkpoint.merge(sdk_contents, &read_mask);
    }

    Ok(GetCheckpointResponse::new(checkpoint))
}
