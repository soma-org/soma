// Copyright (c) Soma Contributors
// SPDX-License-Identifier: Apache-2.0

//! ListTargets RPC handler - lists targets with optional filtering by status and epoch.
//!
//! This implementation uses an index-based approach for efficient target listing.
//! The RpcIndexes trait provides a targets_iter method that iterates through Target objects.

use bytes::Bytes;
use prost::Message;
use prost_types::FieldMask;
use types::base::SomaAddress;
use types::storage::read_store::TargetInfo;

use crate::api::RpcService;
use crate::api::error::{Result, RpcError};
use crate::proto::google::rpc::bad_request::FieldViolation;
use crate::proto::soma::target::target_to_proto_with_id;
use crate::proto::soma::{ErrorReason, ListTargetsRequest, ListTargetsResponse, Target};
use crate::utils::field::{FieldMaskTree, FieldMaskUtil};

const MAX_PAGE_SIZE: usize = 1000;
const DEFAULT_PAGE_SIZE: usize = 50;
const MAX_PAGE_SIZE_BYTES: usize = 512 * 1024; // 512KiB

/// Default fields to return if no read_mask is specified.
pub const READ_MASK_DEFAULT: &str = "id,status,generation_epoch,reward_pool";

#[tracing::instrument(skip(service))]
pub fn list_targets(
    service: &RpcService,
    request: ListTargetsRequest,
) -> Result<ListTargetsResponse> {
    // Get current epoch for computing "expired" status
    let current_epoch =
        service.reader.inner().get_latest_checkpoint().map(|cp| cp.epoch()).unwrap_or(0);

    // Validate status filter if provided
    // Virtual statuses computed server-side:
    // - "expired": Open targets with generation_epoch < current_epoch
    // - "claimable": expired open targets + filled targets past audit window (current_epoch > fill_epoch + 1)
    let status_filter_lower = request.status_filter.as_ref().map(|s| s.to_lowercase());
    let is_expired_filter = status_filter_lower.as_deref() == Some("expired");
    let is_claimable_filter = status_filter_lower.as_deref() == Some("claimable");

    let status_filter = request
        .status_filter
        .as_ref()
        .map(|s| match s.to_lowercase().as_str() {
            "open" | "filled" | "claimed" | "expired" | "claimable" => Ok(s.to_lowercase()),
            _ => Err(FieldViolation::new("status_filter")
                .with_description(format!(
                    "invalid status_filter: '{}'. Must be 'open', 'filled', 'claimed', 'expired', or 'claimable'",
                    s
                ))
                .with_reason(ErrorReason::FieldInvalid)),
        })
        .transpose()?;

    let epoch_filter = request.epoch_filter;

    let submitter_filter = request
        .submitter_filter
        .as_ref()
        .map(|s| {
            s.parse::<SomaAddress>().map_err(|e| {
                FieldViolation::new("submitter_filter")
                    .with_description(format!("invalid submitter_filter: {e}"))
                    .with_reason(ErrorReason::FieldInvalid)
            })
        })
        .transpose()?;

    let page_size = request
        .page_size
        .map(|s| (s as usize).clamp(1, MAX_PAGE_SIZE))
        .unwrap_or(DEFAULT_PAGE_SIZE);

    let page_token =
        request.page_token.as_ref().map(|token| decode_page_token(token)).transpose()?;

    // Validate page token parameters match request
    if let Some(ref token) = page_token {
        if token.status_filter != status_filter
            || token.epoch_filter != epoch_filter
            || token.submitter_filter != request.submitter_filter
        {
            return Err(FieldViolation::new("page_token")
                .with_description("page_token parameters do not match request filters")
                .with_reason(ErrorReason::FieldInvalid)
                .into());
        }
    }

    // Validate and build field mask
    let read_mask = {
        let read_mask = request.read_mask.unwrap_or_else(|| FieldMask::from_str(READ_MASK_DEFAULT));
        read_mask.validate::<Target>().map_err(|path| {
            FieldViolation::new("read_mask")
                .with_description(format!("invalid read_mask path: {path}"))
                .with_reason(ErrorReason::FieldInvalid)
        })?;
        FieldMaskTree::from(read_mask)
    };

    // Get indexes for target iteration
    let indexes = service.reader.inner().indexes().ok_or_else(RpcError::not_found)?;

    // Get the cursor from page token if present
    let cursor = page_token.as_ref().map(|t| t.cursor.clone());

    // Map virtual status filters to DB queries:
    // - "expired" → query "open", post-filter by generation_epoch < current_epoch
    // - "claimable" → query all (no DB filter), post-filter for both expired open + filled past audit
    // - "open" → query "open", post-filter to exclude expired
    let db_status_filter = if is_expired_filter {
        Some("open".to_string())
    } else if is_claimable_filter {
        None // need both open + filled, so no DB-level status filter
    } else {
        status_filter.clone()
    };

    // Iterate through targets using the index
    let mut iter = indexes
        .targets_iter(db_status_filter, epoch_filter, cursor)
        .map_err(|e| RpcError::new(tonic::Code::Internal, e.to_string()))?;

    let mut targets = Vec::with_capacity(page_size);
    let mut size_bytes = 0;

    while let Some(target_info) =
        iter.next().transpose().map_err(|e| RpcError::new(tonic::Code::Internal, e.to_string()))?
    {
        // Load the full target object
        let Some(object) =
            service.reader.inner().get_object_by_key(&target_info.target_id, target_info.version)
        else {
            tracing::debug!(
                "unable to find target {}:{} while iterating",
                target_info.target_id,
                target_info.version.value()
            );
            continue;
        };

        // Deserialize the target
        let target: types::target::TargetV1 = match bcs::from_bytes(object.data.contents()) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("failed to deserialize target {}: {}", target_info.target_id, e);
                continue;
            }
        };

        // Compute virtual statuses
        let is_expired = matches!(target.status, types::target::TargetStatus::Open)
            && target.generation_epoch < current_epoch;
        let is_filled_claimable = matches!(target.status, types::target::TargetStatus::Filled { fill_epoch } if current_epoch > fill_epoch + 1);
        let is_claimable = is_expired || is_filled_claimable;

        // Apply post-filters for virtual statuses
        if is_expired_filter && !is_expired {
            continue; // "expired" filter: skip non-expired targets
        }
        if is_claimable_filter && !is_claimable {
            continue; // "claimable" filter: skip non-claimable targets
        }
        // Apply submitter filter
        if let Some(ref addr) = submitter_filter {
            if target.submitter.as_ref() != Some(addr) {
                continue;
            }
        }
        if status_filter.as_deref() == Some("open") && is_expired {
            continue; // "open" filter: skip expired targets (only show truly open)
        }

        // Convert to proto
        let mut target_proto = target_to_proto_with_id(&target_info.target_id, &target, &read_mask);

        // Override status for virtual statuses
        if is_claimable && target_proto.status.is_some() {
            target_proto.status = Some("claimable".to_string());
        } else if is_expired && target_proto.status.is_some() {
            target_proto.status = Some("expired".to_string());
        }

        size_bytes += target_proto.encoded_len();
        targets.push(target_proto);

        if targets.len() >= page_size || size_bytes >= MAX_PAGE_SIZE_BYTES {
            break;
        }
    }

    // Build next page token if there are more results
    let next_page_token = iter
        .next()
        .transpose()
        .map_err(|e| RpcError::new(tonic::Code::Internal, e.to_string()))?
        .map(|cursor_info| {
            encode_page_token(PageToken {
                status_filter: status_filter.clone(),
                epoch_filter,
                submitter_filter: request.submitter_filter.clone(),
                cursor: cursor_info,
            })
        });

    let response = ListTargetsResponse { targets, next_page_token, ..Default::default() };
    Ok(response)
}

/// Page token for ListTargets pagination.
#[derive(serde::Serialize, serde::Deserialize)]
struct PageToken {
    status_filter: Option<String>,
    epoch_filter: Option<u64>,
    submitter_filter: Option<String>,
    cursor: TargetInfo,
}

fn decode_page_token(page_token: &[u8]) -> Result<PageToken> {
    bcs::from_bytes(page_token).map_err(|_| {
        FieldViolation::new("page_token")
            .with_description("invalid page_token")
            .with_reason(ErrorReason::FieldInvalid)
            .into()
    })
}

fn encode_page_token(page_token: PageToken) -> Bytes {
    bcs::to_bytes(&page_token).unwrap().into()
}
